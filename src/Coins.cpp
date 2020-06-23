#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

#define BUILD_DICTIONARY true
#define BUILD_SVM true
#define DICTIONARY_SAMPLES 20
#define DICTIONARY_SIZE 1000

//Função auxiliar responsável por retornar o tempo atual
string getNow(){
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time (&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer,80,"%d-%m-%Y %I:%M:%S",timeinfo);
	string timeString(buffer);

	return timeString;
}

//Dado um valor de classe c, retorna classificação textual da moeda
string getCoinValue(int c){
	switch (c){
		case 0: return "R$0.01 (anverso)";
		case 1: return "R$0.05 (anverso)";
		case 2: return "R$0.10 (anverso)";
		case 3: return "R$0.25 (anverso)";
		case 4: return "R$0.50 (anverso)";
		case 5: return "R$1.00 (anverso)";
		case 10: return "R$0.01 (reverso)";
		case 11: return "R$0.05 (reverso)";
		case 12: return "R$0.10 (reverso)";
		case 13: return "R$0.25 (reverso)";
		case 14: return "R$0.50 (reverso)";
		case 15: return "R$1.00 (reverso)";
		default: return "Indefinido";
	}
}


//Leitura de imagens
Mat readImage(string path){
	Mat image;
	image = imread(path);
	if(!image.data){
		exit(-1);
	}
	return image;
}

//Dada uma imagem de entrada, retorna sua versão em escala-de-cinza
Mat colorToGray(Mat input){
	Mat output;
	cvtColor(input, output, CV_BGR2GRAY);
	return output;
}

//Dada uma imagem de entrada e um tamanho de mascára d, retorna uma versão suavizada da imagem
Mat smoothFilter(Mat input, int d){
	Mat output;
	medianBlur(input, output, d);
	return output;
}

//Dada uma imagem de entrada, retorna máscara contendo o maior círculo detectado através da função HoughCircles
Mat extractHoughCircleMask(Mat input){
	vector<Vec3f> circles;
	Mat inputFilterGray = smoothFilter(colorToGray(input), 21);

	HoughCircles(inputFilterGray, circles, CV_HOUGH_GRADIENT, 2, input.rows, 100, 100, 0, input.rows);
	if(circles.size() == 0){
		exit(-1);
	}

	Mat mask(input.size(), CV_8UC1);
	mask.setTo(0);

	Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
	int radius = cvRound(circles[0][2]);
	radius = radius-(radius*10/100);
	circle(mask, center, radius, 1 ,-1, 8, 0);

	return mask;
}


//Dada uma imagem e uma máscara, extrai descritores (utilizando o método Sift)
Mat extractDescriptorsUsingSift(Mat input, Mat mask){
	Mat inputFilterGray = smoothFilter(colorToGray(input), 5);

	SiftFeatureDetector detector(0);
	vector<KeyPoint> keypoints;
	detector.detect(inputFilterGray, keypoints, mask);

	Mat descriptors;
	SiftDescriptorExtractor descriptorExtractor;
	descriptorExtractor.compute(inputFilterGray, keypoints, descriptors);

	return descriptors;
}

//Dada uma imagem, uma máscara e um extrator de descritores BOW, extrai descritores utilizando este descritor
Mat extractDescriptorsUsingBOW(Mat input, Mat mask, BOWImgDescriptorExtractor descriptorExtractor){
	Mat inputFilterGray = smoothFilter(colorToGray(input), 5);

	SiftFeatureDetector detector(0);
	vector<KeyPoint> keypoints;
	detector.detect(inputFilterGray, keypoints, mask);

	Mat descriptors;
	descriptorExtractor.compute(inputFilterGray, keypoints, descriptors);

	return descriptors;
}

//Função responsável por criar um dicionário de características e salva-lo em path
void buildDictionary(string path, int dictionarySize){
	printf("%s Iniciando criação do dicionário amostrando %d imagens de cada moeda.\n", getNow().data(), DICTIONARY_SAMPLES);

	Mat descritoresMat;
	for(int c=0;c<6;c++){
		for(int l=0; l<2; l++){
			for(int i=0;i<DICTIONARY_SAMPLES;i++){
				char fileName[50];
				sprintf(fileName, "%s%d%s%d%s%d%s", "train/", c, "/", l, "/" ,i, ".png");
				Mat input = readImage(fileName);
				Mat mask = extractHoughCircleMask(input);
				Mat descriptors = extractDescriptorsUsingSift(input, mask);
				descritoresMat.push_back(descriptors);
			}
		}
	}

	printf("%s Construindo dicionário com %d palavras em %s... \n", getNow().data(), dictionarySize, path.data());

	BOWKMeansTrainer bowTrainer(dictionarySize);
	Mat dictionary=bowTrainer.cluster(descritoresMat);

	FileStorage fs(path, FileStorage::WRITE);
	fs << "coinDictionary" << dictionary;
	fs.release();

	printf("%s Dicionário construído com sucesso!\n", getNow().data());
}

//Função responsável por treinar uma SVM e salva-la em path
void trainSVM(string path, BOWImgDescriptorExtractor descriptorExtractor){
	printf("%s Iniciando dados do SVM amostrando %d imagens de cada moeda.\n", getNow().data(), DICTIONARY_SAMPLES);

	Mat trainingData;
	Mat labels;

	for(int c=0;c<6;c++){
		for(int l=0; l<2; l++){
			for(int i=0;i<DICTIONARY_SAMPLES;i++){
				char fileName[50];
				sprintf(fileName, "%s%d%s%d%s%d%s", "train/", c, "/", l, "/" ,i, ".png");
				Mat input = readImage(fileName);
				Mat mask = extractHoughCircleMask(input);
				Mat bowDescriptors = extractDescriptorsUsingBOW(input, mask, descriptorExtractor);

				trainingData.push_back(bowDescriptors);
				int label = c + l*10;
				labels.push_back(label);
			}
		}
	}


	printf("%s Treinando o SVM e salvando em %s.\n", getNow().data(), path.data());

	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	CvSVM svm;
	svm.train(trainingData, labels, Mat(), Mat(), params);
	svm.save(path.data());

	printf("%s SVM construído com sucesso!\n",getNow().data());
}

//Função que realiza a leitura de um dicionário de características
Mat readDictionary(string path){
	Mat dictionary;
	FileStorage fs(path, FileStorage::READ);
	fs["coinDictionary"] >> dictionary;
	fs.release();
	return dictionary;
}


int main(int argc, char *argv[]){
	//Fase 1. Criação do dicionário de características
	if(BUILD_DICTIONARY){
		buildDictionary("coinDictionary.yml", DICTIONARY_SIZE);
	}
  Mat dictionary = readDictionary("coinDictionary.yml");

	//Fase 2. Criação e treinamento do SVM com base no descritor criado
	BOWImgDescriptorExtractor descriptorExtractor(new SiftDescriptorExtractor(), new FlannBasedMatcher());
	descriptorExtractor.setVocabulary(dictionary);
	if(BUILD_SVM){
		trainSVM("coinSVM.yml", descriptorExtractor);
	}
	CvSVM svm;
	svm.load("coinSVM.yml");

	//Fase 3. Predição de valores de moedas
	//exemplo de entrada
	int cTest=5; //moeda de 1 real
	int lTest=1; //lado reverso
	int id=35; //identificador unico da imagem no banco de imagens

	printf("%s Testando moeda: %s\n",getNow().data(), getCoinValue(cTest).data());
	char fileName[50];
	sprintf(fileName, "%s%d%s%d%s%d%s", "train/", cTest, "/", lTest, "/" , id, ".png");

	Mat inputTest = readImage(fileName);
	Mat maskTest = extractHoughCircleMask(inputTest);
	Mat bowDescriptorsTest = extractDescriptorsUsingBOW(inputTest, maskTest, descriptorExtractor);
	float response = svm.predict(bowDescriptorsTest);
	printf("%s Resposta obtida: %s - ", getNow().data(), getCoinValue((int)response).data());
	if((int)response == cTest+lTest*10) {
		return 1; //sucesso
	}
	return 0; //falha
}
