/*
 * Text-Recog.cpp
 *
 * A demo program of End-to-end Scene Text Detection and Recognition using webcam or video.
 *
 * Created on: Jul 31, 2014
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/text.hpp"

#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::text;

Mat src, src_gray, src_binary, res , mask, subtraction;
int dilate_size = 5;

/**
* @function main
*/
int main(int argc, char** argv)
{
	/// Load source image
	String imageName("../señal.jpg"); // by default
	if (argc > 1)
	{
		imageName = argv[1];
	}
	src = imread(imageName, IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "No image supplied ..." << endl;
		return -1;
	}

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	medianBlur(src_gray, src_gray, 5);
	imshow("Source", src);
	waitKey(-1);

	threshold(src_gray, src_binary,80 , 255, THRESH_OTSU);
	imshow("Threshold", src_binary);

	ximgproc::niBlackThreshold(src_gray, src_binary, 255, THRESH_OTSU, 5, 0.2, ximgproc::LocalBinarizationMethods::BINARIZATION_SAUVOLA);
	bitwise_not(src_binary, src_binary);

	// Since MORPH_X : 2,3,4,5 and 6
	Mat element = getStructuringElement(1, Size(2 * dilate_size + 1, 2 * dilate_size + 1), Point(dilate_size, dilate_size));

	/// Apply the specified morphology operation
	morphologyEx(src_binary, mask, MORPH_DILATE, element);
	waitKey(-1);

	Mat background = Mat::zeros(src.size(), src.type());

	for (size_t i = 0; i < background.rows; i++)
	{
		for (size_t j = 0; j < background.cols; j++)
		{
			if (src_binary.at<uchar>(i, j) == 0)
				background.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	}

	inpaint(background, mask, res, 3, INPAINT_TELEA);
	cvtColor(res, res, COLOR_BGR2GRAY);
	imshow("Background", res);
	waitKey(-1);

	subtraction = res - src_gray;
	threshold(subtraction, src_binary, 80, 255, THRESH_OTSU);

	bitwise_not(src_binary, src_binary);
	imshow("Binary/Inverted", src_binary);
	waitKey(-1);
}



