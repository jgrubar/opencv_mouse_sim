#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;


//funkcije
void draw_refernce_area(Mat frame_out,bool iter);
Mat hand_histogram(Mat frame);
Mat mask_with_hist(Mat frame,Mat hist);
void opening_operation(Mat img, int kernel, Point kernel_size);
Mat get_hand_contour(Mat img);
void remove_face(Mat input, Mat output);
Mat detect_fingers(Mat input, Mat frame);

//globalne spremenljivke
vector<Rect>sample_rects;
Mat vsi_vzorci;
CascadeClassifier face_cascade;

int main(){
	
	VideoCapture cap(0);
	//cap.set(CAP_PROP_FRAME_WIDTH,640);
	//cap.set(CAP_PROP_FRAME_HEIGHT,480);
	
	face_cascade.load("haarcascade_frontalface_alt.xml");

	if(!cap.isOpened()){
		cout << "can't open camera" << endl;
	}
	bool vzorec = false;
	Mat frame,frame_out,hist,hand_mask,contours;
	bool prvic=true;
	while(true){
		cap >> frame;
		frame_out=frame.clone();
		draw_refernce_area(frame_out,prvic);
		prvic=false;
		imshow("kek",frame_out);
		if(vzorec){
			hand_mask = mask_with_hist(frame,hist);
			imshow("HISTOGRAM",hist);
			remove_face(frame,hand_mask);
			imshow("handmask",hand_mask);
			contours = get_hand_contour(hand_mask);
			imshow("contours",contours);
		}
		int key=waitKey(10);
		if(key==115){
			//poberi vzorce, izracunaj histogram
			//za nov vzorec brisi prejsnjega
			/*
			while(vsi_vzorci.rows>0){
				vsi_vzorci.pop_back();
			}
			*/
			hist = hand_histogram(frame);
			vzorec=true;

			
		}
		//pobrisi vzorec za histogram
		if(key==100){
			while(vsi_vzorci.rows>0){
				vsi_vzorci.pop_back();
			}

		}
		if(key==113){
			break;
		}
	}

	return 0;
}
void opening_operation(Mat img, int kernel, Point kernel_size){
	Mat structuring_element=getStructuringElement(kernel,kernel_size);
	morphologyEx(img,img,MORPH_OPEN,structuring_element);
}

void draw_refernce_area(Mat frame_out,bool prvic){
	int start_x = frame_out.size().width/6;
	int start_y = frame_out.size().height/2;
	int reference_x [] = {start_x,start_x+30,start_x+60};
	int reference_y [] = {start_y,start_y+50,start_y+100};
	
	Mat frame = frame_out.clone();	
	//vector<Rect>sample_rects;
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			if(prvic){
				sample_rects.push_back(Rect(reference_x[i], reference_y[j], 20, 20));
			}
			rectangle(frame_out,Rect(reference_x[i], reference_y[j], 20, 20),Scalar(255,0,0));
		}
	}
	/*
	cout << "izpisujem kvadrate" << endl;
	for(int i=0;i<sample_rects.size();i++){
		cout << sample_rects[i] << endl;
	}
	*/

	/*
	Mat vsi_vzorci;
	Mat frame_hsv;
	cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
	for(int i=0;i<sample_rects.size();i++){
		Mat sample = Mat(frame_hsv,sample_rects[i]);	
		vsi_vzorci.push_back(sample);
	}
	cout << "WIDTH: " << vsi_vzorci.size().width << "HEIGHT: " << vsi_vzorci.size().height << endl;
	imshow("SKUPEK",vsi_vzorci);
	//vsi_vzorci.release();
	*/
	
}
Mat hand_histogram(Mat frame){
	vsi_vzorci;
	Mat frame_hsv;
	cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
	for(int i=0;i<sample_rects.size();i++){
		Mat sample = Mat(frame_hsv,sample_rects[i]);	
		vsi_vzorci.push_back(sample);
	}
	//cout << "WIDTH: " << vsi_vzorci.size().width << "HEIGHT: " << vsi_vzorci.size().height << endl;
	cout << "VZORCI: " << vsi_vzorci.size() << endl;
	//vsi_vzorci.release();

	//izracunaj histogram
	int channels [] = {0,1};
	Mat hist;
	int hbins = 180;
	int sbins = 256;
	int hist_size [] = {hbins,sbins};
	float hranges [] = {0,180};
	float sranges [] = {0,256};
	const float* ranges [] = {hranges,sranges};
	calcHist(&vsi_vzorci,1,channels,Mat(),hist,2,hist_size,ranges,true,false);	
	normalize(hist,hist,0,255,NORM_MINMAX,-1,Mat());
	return hist;

}


Mat mask_with_hist(Mat frame,Mat hist){
	Mat frame_hsv,mask;
	cvtColor(frame,frame_hsv,COLOR_BGR2HSV);


	int channels [] = {0,1};
	int hbins = 180;
	int sbins = 256;
	int hist_size [] = {hbins,sbins};
	float hranges [] = {0,180};
	float sranges [] = {0,256};
	const float* ranges [] = {hranges,sranges};

	calcBackProject(&frame_hsv,1,channels,hist,mask,ranges,1);
	
	//imshow("calcback",mask);
	
	Mat kernel  = getStructuringElement(MORPH_ELLIPSE,{15,15});
	filter2D(mask,mask,-1,kernel,Point(-1,-1));
	Mat t;
	threshold(mask,t,150,255,THRESH_BINARY);

	//imshow("pred erode",t);
	opening_operation(t,MORPH_OPEN,{15,15});
	erode(t,t,Mat(),Point(-1,-1),6);
	//imshow("po odpiranju THRESH",t);
	
	Mat mt;
	cvtColor(t,mt,COLOR_GRAY2BGR);	

	Mat x;
	bitwise_and(frame,mt,x);
	//imshow("X",x);
	//barvna slika z masko
	//return frame;
	
	return t;
}
Mat get_hand_contour(Mat img){
	int thresh=100;	
	Mat contours_img=Mat::zeros(img.size(),CV_8UC3);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	/*
	for(size_t i=0; i<contours.size(); i++){
		drawContours(contours_img, contours, (int)i, Scalar(255,0,0), 2, LINE_8, hierarchy, 0);
	}
	*/
	int biggest_contour_index = -1;
	double biggest_area = 0.0;

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i], false);
		if (area > biggest_area) {
			biggest_area = area;
			biggest_contour_index = i;
		}
	}

	if (biggest_contour_index < 0){
		return contours_img;
	}
	vector<Point> hull_points;
	vector<int> hull_ints;

	convexHull(Mat(contours[biggest_contour_index]), hull_points, true);
	convexHull(Mat(contours[biggest_contour_index]), hull_ints, false);

	//defekti
	vector<Vec4i> defects;
	convexityDefects(Mat(contours[biggest_contour_index]), hull_ints, defects);
	
	//bounding box za center
	Rect bounding_box = boundingRect(hull_points);
  	rectangle(contours_img, bounding_box, Scalar(255, 0, 0));
  	Point center = Point(bounding_box.x + bounding_box.width / 2, bounding_box.y + bounding_box.height / 2);

	for (size_t i = 0; i < defects.size(); i++){
		Point p1 = contours[biggest_contour_index][defects[i][0]];
		Point p2 = contours[biggest_contour_index][defects[i][1]];
		Point p3 = contours[biggest_contour_index][defects[i][2]];
		line(contours_img, p1, p3, Scalar(0, 255, 0), 2);
		line(contours_img, p3, p2, Scalar(0, 255, 0), 2);
	}
	cout << "defekti" << endl;
	
	for(int i=0;i<defects.size();i++){
		circle(contours_img,contours[biggest_contour_index][defects[i][0]],20,Scalar(0,0,255),1);
	}

	
	drawContours(contours_img, contours, biggest_contour_index, Scalar(255,0,0), 2, LINE_8, hierarchy, 0);
	//drawContours(contours_img, vector<vector<Point>> {hull_points}, 0, Scalar(128), 3, 8);
	drawContours(contours_img, vector<vector<Point>> {hull_points}, 0, Scalar(0,0,255), 3, 8);

	return contours_img;
}
void remove_face(Mat input, Mat output){
	vector<Rect> faces;
	Mat frameGray;

	cvtColor(input, frameGray, COLOR_BGR2GRAY);
	equalizeHist(frameGray, frameGray);

	//face_cascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));
	face_cascade.detectMultiScale(frameGray, faces, 1.1, 2, 0, Size(120, 120));

	for (size_t i = 0; i < faces.size(); i++) {
		rectangle(
			output,
			Point(faces[i].x, faces[i].y),
			Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
			Scalar(0, 255, 0),
			-1
		);
	}
	//imshow("OUT",output);
}
