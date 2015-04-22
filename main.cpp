//****************************************************************************************************
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>


#include <pcl\io\io.h>
#include <pcl\io\pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl\point_types.h>
#include <pcl\ros\conversions.h>


#include <iostream>
#include <ctime>
#include <fstream>
#include <cmath>
#include <iomanip>

//#include "opencv2/flann/flann.hpp" 

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2\core\cuda_devptrs.hpp"

#include "cv_costum.h"
#include "util.h"
#include "FGObject.h"
#include "FGExtraction.h"
#include "stereo.h"

using namespace pcl;
using namespace cv;
using namespace std;;
using namespace pcl::io;
using namespace pcl::console;


const char* const path="C:\\Users\\qiuyu\\Desktop\\rb trout";

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> View1 (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) 
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> View2 (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                                            pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud2)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->initCameraParameters ();

  int v1(0);
  viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  viewer->setBackgroundColor (0, 0, 0, v1);
  viewer->addText("Point Cloud1", 10, 10, "v1 text", v1);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "sample cloud1", v1);

  int v2(0);
  viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
  viewer->addText("Point Cloud2", 10, 10, "v2 text", v2);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud2, "sample cloud2", v2);


  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud2");
  viewer->addCoordinateSystem (1.0);

  return (viewer);
}



//void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
//                         void* viewer_void)
//{
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
//  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
//      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
//  {
//    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;
//
//    char str[512];
//    sprintf (str, "text#%03d", text_id ++);
//    viewer->addText ("clicked here", event.getX (), event.getY (), str);
//  }
//}
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ()
//{
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  viewer->addCoordinateSystem (1.0);
//
//  viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
//  viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);
//
//  return (viewer);
//}

// crop the target from input image and rotate to horizontal orientation
Rect cropTargetImage(const FGObject& obj, InputArray src, InputArray fgSrc, OutputArray dst, OutputArray dstFg)
{
	if(!src.obj || !fgSrc.obj) return Rect();
	Mat inImg = src.getMat();
	Mat fgImg = fgSrc.getMat();

	float x_min = imageWidth, y_min = imageHeight, x_max = 0, y_max = 0;
	for(int k = 0; k < 4; ++k){
		if(obj.uPoints[k].x < x_min) x_min = obj.uPoints[k].x;
		if(obj.uPoints[k].y < y_min) y_min = obj.uPoints[k].y;
		if(obj.uPoints[k].x > x_max) x_max = obj.uPoints[k].x;
		if(obj.uPoints[k].y > y_max) y_max = obj.uPoints[k].y;
	}
	Point2f tl (x_min, y_min);
	Point2f br (x_max, y_max);
//???
	tl += 0.8*(tl - obj.uCenter);
	br += 0.8*(br - obj.uCenter);
	tl.x = tl.x < 0 ? 0 : (tl.x > imageWidth-1 ? imageWidth-1 : tl.x);
	tl.y = tl.y < 0 ? 0 : (tl.y > imageHeight-1 ? imageHeight-1 : tl.y);
	br.x = br.x < 0 ? 0 : (br.x > imageWidth-1 ? imageWidth-1 : br.x);
	br.y = br.x < 0 ? 0 : (br.y > imageHeight-1 ? imageHeight-1 : br.y);

	Rect roiRect (tl, br);
	Mat roiImg = inImg(roiRect);
	Mat roiFgImg = fgImg(roiRect);
	//showImage("roiImg", roiImg, 1, 1);
	//showImage("roiFgImg", roiFgImg, 1);


	// preserve only the blob with largest area
	vector<vector<Point>> contours = extractContours(roiFgImg);
	double maxA = 0;
	int argMaxA = 0;
	for(int n = 0; n < contours.size(); ++n){
		double area = contourArea(contours[n]);
		if(area > maxA) {
			maxA = area;
			argMaxA = n;
		}
	}
	//showImage("old fg", targetFgImg, 1, 1);

	Mat tempFg = Mat::zeros(roiFgImg.size(), roiFgImg.type());
	drawContours(tempFg, contours, argMaxA, Scalar(255), -1);
	roiFgImg = tempFg;

	//showImage("new fg", targetFgImg, 1, 0);


	// rotate target by angle of bounding box
	Point2f roiCenter = obj.uCenter - tl;
	double angle = obj.angle;
	//cout << "angle = " << angle << endl;
	if(angle >= 90){
		angle = obj.angle - 180;
	}

	Mat R = getRotationMatrix2D(roiCenter, angle, 1.0); 

	Mat rotatedRoiImg, rotatedRoiFgImg;
	warpAffine(roiImg, rotatedRoiImg, R, roiImg.size());
	warpAffine(roiFgImg, rotatedRoiFgImg, R, roiFgImg.size());

	//showImage("rotatedRoiImg", rotatedRoiImg, 1, 1);
	//showImage("rotatedRoiFgImg", rotatedRoiFgImg, 1);

	tl.x = roiCenter.x - 0.7 * obj.uWidth;
	tl.y = roiCenter.y - 0.9 * obj.uHeight;
	br.x = roiCenter.x + 0.7 * obj.uWidth;
	br.y = roiCenter.y + 0.9 * obj.uHeight;
	tl.x = tl.x < 0 ? 0 : (tl.x > rotatedRoiImg.cols-1 ? rotatedRoiImg.cols-1 : tl.x);
	tl.y = tl.y < 0 ? 0 : (tl.y > rotatedRoiImg.rows-1 ? rotatedRoiImg.rows-1 : tl.y);
	br.x = br.x < 0 ? 0 : (br.x > rotatedRoiImg.cols-1 ? rotatedRoiImg.cols-1 : br.x);
	br.y = br.x < 0 ? 0 : (br.y > rotatedRoiImg.rows-1 ? rotatedRoiImg.rows-1 : br.y);
	roiRect = Rect(tl, br);

	Mat croppedRoiImg = rotatedRoiImg(roiRect);
	Mat croppedRoiFgImg = rotatedRoiFgImg(roiRect);

	//showImage("croppedRoiImg", croppedRoiImg, 1, 1);
	//showImage("croppedRoiFgImg", croppedRoiFgImg, 1);


	dst.create(croppedRoiImg.size(), croppedRoiImg.type());
	Mat targetImg = dst.getMat();
	dstFg.create(croppedRoiFgImg.size(), croppedRoiFgImg.type());
	Mat targetFgImg = dstFg.getMat();

	croppedRoiImg.copyTo(targetImg);
	croppedRoiFgImg.copyTo(targetFgImg);

	return roiRect;
}

void plotTarget(Mat drawing, FGObject target)
{
	//char labelText[5];

	// draw the oriented bounding box
	for(int j = 0; j < 4; ++j){
		line(drawing, target.umPoints[j], target.umPoints[(j+1)%4], target.rectColor, 4, 8);
		//sprintf_s(labelText, "%1d", j); //,target.rPoints[j].x,target.rPoints[j].y);
		//putText(drawing, string(labelText), target.rmPoints[j], FONT_HERSHEY_PLAIN, 3, color, 4);
	}
	// plot the center point
	circle(drawing, target.umCenter, 6, target.rectColor, 3);
	
	// plot the 2 endpoints
	circle(drawing, target.umlMidpoint, 6, Scalar(0, 255, 255), -1);
	circle(drawing, target.umrMidpoint, 6, Scalar(0, 255, 255), -1);

	// put the tracking number over the target
	if(target.trackingNum > 0){
		char numText[5];
		sprintf_s(numText, "%d", target.trackingNum);
		putText(drawing, string(numText), target.umCenter, FONT_HERSHEY_PLAIN, 5, Scalar(0, 0, 192), 7);
	}
}

// applies foreground segmentation and stereo matching
// produces the disparity map
int mainSegAndStereo()
{
	char	filename[256];
	Mat		srcLeft, fgLeft;
	Mat		srcRight, fgRight;
	Mat		tempSrcLeft, tempFgLeft;
	Mat		tempSrcRight, tempFgRight;
	
	int		outWidth = 2048;
	int		outHeight = 1024;
	
	int		thresh = 48;
	int		seLength = 9;
	double	minArea = 1000;
	double	maxArea = 1e6;
	double	minAspRatio = 1.8;
	double	maxAspRatio = 8.0;
	
	// start and end image numbers for fish tracking
	// 1st sequence
	//int		startImg = 11836, endImg = 11877;
	// 2nd sequence
	//int		startImg = 10652, endImg = 10711;
	// 3rd sequence
	int			startImg = 10807, endImg = 10809;
//	int			startImg = 10792, endImg = 10796;
	// 4th sequence for tracking
	//int		startImg = 6323, endImg = 6380;
	// 5th sequence for tracking
	//int		startImg = 16690, endImg = 16719;

	// segmented objects
	vector<FGObject>*		  leftObjects;
	vector<FGObject>*		  rightObjects;
	vector<FGObject>*		  prevLeftObjects = NULL;
	vector<FGObject>*		  prevRightObjects = NULL;
	vector<vector<Point3f> >  objectPoints;
    vector<vector<Point2f> >  imagePoints[2];


	//static void
 //   StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated=true, bool showRectified=true)
 //   {
 //   if( imagelist.size() % 2 != 0 )
 //   {
 //       cout << "Error: the image list contains odd (non-even) number of elements\n";
 //       return;
 //   }

 //   bool displayCorners = false;//true;
 //   const int maxScale = 2;
 //   const float squareSize = 1.f;  // Set this to your actual square size

	//vector<StereoObject*>	prevStereoObjects;
	//vector<StereoObject*>	trackedObjects;

	// create objects for each component of our algorithm
	FGExtraction fgExtractorLeft = FGExtraction();
	FGExtraction fgExtractorRight = FGExtraction();

	// stereo vision
	Stereo stereoMatcher = Stereo(imageWidth, imageHeight);
	stereoMatcher._minArea = minArea;
	stereoMatcher._maxArea = maxArea;
	if (!stereoMatcher.loadCameraParameters("camera_param.xml"))
		return -1;
	
	// create a file to store the lengths
	sprintf_s(filename, "%s\\length.csv", model);
	ofstream foutLen(filename, ios::out);
	
	// create a file to store the center (as the ground truth)
	sprintf_s(filename, "%s\\fishCenters_10787_10846.txt", model);
	ofstream foutCen(filename, ios::out);
	// process through images
	for (int im = startImg; im < endImg+1; im++)
	{
		cout << "Reading image " << im << endl;
		
		//  read in the left source image
		sprintf_s(filename, "%s\\%05d.jpg", leftImgPath, im);
		srcLeft = imread(filename, 0);
		maxArea = srcLeft.size().area();

		//  read in the right source image
		sprintf_s(filename, "%s\\%05d.jpg", rightImgPath, im);
		srcRight = imread(filename, 0);

		//  extract FG targets in the left image
		leftObjects = fgExtractorLeft.extractFGTargets(srcLeft, fgLeft, seLength, thresh, minArea, maxArea, minAspRatio, maxAspRatio);

        //  check if we have any targets
		if (!leftObjects->empty()) {
			//  left image has some targets - process the right image
			//  extract FG objects
			rightObjects = fgExtractorRight.extractFGTargets(srcRight, fgRight, seLength, thresh, minArea, maxArea, minAspRatio, maxAspRatio);
			
			//  check if we have any objects in our right image
			if (!rightObjects->empty()) {
				//  we have objects - process stereo rectification first
				bool isRecified = stereoMatcher.stereoRectification(srcLeft, srcRight, fgLeft, fgRight);
				// stereo matching for left and right targets
				stereoMatcher.findEpipolarCorrespondence(leftObjects, rightObjects);
				

				// write all object center points to file
				foutCen << im << ' ';
				for(size_t i = 0; i < leftObjects->size(); ++i){
					FGObject* obj = &(*leftObjects)[i];
					if(obj->stereoMatch){
						FGObject* objRight = obj->stereoMatch;
						foutCen << i+1 << ':'
							    << int(obj->uCenter.x) << ',' << int(obj->uCenter.y) << ',' 
							    << int(objRight->uCenter.x) << ',' << int(objRight->uCenter.y) << ' ';
					}
				}
				foutCen << endl;

				// write object length to file
				for(size_t i = 0; i < leftObjects->size(); ++i){
					FGObject* obj = &(*leftObjects)[i];
					if(!obj->triMidpoints.empty()){
						double len = norm(obj->triMidpoints[0] - obj->triMidpoints[1]);
						foutLen << im << ',' << len << ',' << obj->triMidpoints[0] << ',' << obj->triMidpoints[1] << endl;
					}	
				}



				   cv::Mat_<double> cameraMatrix1(3,3); // 3x3 matrix
                   cv::Mat_<double> distCoeffs1(5,1);   // 5x1 matrix for five distortion coefficients
                   cv::Mat_<double> cameraMatrix2(3,3); // 3x3 matrix
                   cv::Mat_<double> distCoeffs2(5,1);   // 5x1 matrix
                   cv::Mat_<double> R(3,3);             // 3x3 matrix, rotation left to right camera
                   cv::Mat_<double> T(3,1);             // * 3 * x1 matrix, translation left to right proj. center
                  // ^^ that's the main diff of the code, (3,1) instead of (4,1)

				   cameraMatrix1 << 1403.88281045803, 0, 983.259756611241, 0, 1404.16164011044, 972.365997016595, 0, 0, 1;
                   cameraMatrix2 << 1379.21056450551, 0, 1020.33776668983, 0, 1380.60203266949, 987.187110331994, 0, 0, 1;
				   distCoeffs1   << -0.164298734149432,0.0744241568770007, 0.000393790582410684, -0.000812969939495568, 0;
                   distCoeffs2   << -0.164660245562362, 0.0699037277881666, 0.000576543211218478,0.000149723456774887, 0;
				   R   << 0.982506526223805, -0.0229367085573322, 0.184810263048856, 0.0234227678050390, 0.999725549411792, -0.000446991769481422, -0.184749289243511, 0.00476794021006535, 0.982774117928515;
                   T   << -944.602060675234, -18.1971626246256, 86.6474980373243;
               
				   //Mat filteredl = Mat(srcLeft.size(),CV_8U);
       //            Mat filteredr = Mat(srcRight.size(),CV_8U);
		     //      filteredl.setTo(Scalar(128),fgLeft);
				   //filteredr.setTo(Scalar(128),fgRight);
		     //      Mat frame1,frame2;
				   //bilateralFilter(srcLeft,filteredl,0,20.0,2.0);
				   //bilateralFilter(srcRight,filteredr,0,20.0,2.0);
		     //      filteredl.copyTo(frame1, fgLeft);  // Copies non-masked pixels from filtered to frame.
				   //filteredr.copyTo(frame2,fgRight); 				   
				   //cv::Mat disp, disp8;


      //             StereoBM sbm;
      //            
				  // sbm.state->SADWindowSize = 17;
      //             sbm.state->numberOfDisparities = 288;
      //             sbm.state->preFilterSize = 5;
      //             sbm.state->preFilterCap = 61;
      //             sbm.state->minDisparity = 0;
      //             sbm.state->textureThreshold = 50;
      //             sbm.state->uniquenessRatio = 0;
      //             sbm.state->speckleWindowSize = 0;
      //             sbm.state->speckleRange = 8;
      //             sbm.state->disp12MaxDiff = 1;
				  // sbm(frame1, frame2, disp,CV_32F); 

      //            normalize(disp,disp8, 0, 255, NORM_MINMAX);
				  //sprintf_s(filename, "%s\\disp-%05d.jpg", disparity, im);
				  //imwrite(string(filename), disp8);
				   Mat dispMat;
				   stereoMatcher.refineDisparity(leftObjects, rightObjects, dispMat);
				   Mat dispMatNorm = Mat::zeros(dispMat.size(), CV_8U);
				   Mat dispMatNormn = Mat::zeros(dispMat.size(), CV_8U);	
				   Mat R1, R2, P1, P2, Q;
				   Mat mapXLeft, mapYLeft;
				   Mat mapXRight,mapYRight;
                   Mat frame1, frame2; 
				   //import the parameters from xml file. 
				   stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,srcLeft.size(),  R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, srcLeft.size(), 0, 0);
				   initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, srcLeft.size(),CV_32FC1, mapXLeft, mapYLeft);
                   initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, srcLeft.size(), CV_32FC1, mapXRight, mapYRight);
                   
				   cv::remap(srcLeft, dispMatNorm, mapXLeft, mapYLeft, CV_INTER_CUBIC, BORDER_CONSTANT, 0);
                   cv::remap(srcLeft, dispMatNormn, mapXRight, mapYRight, CV_INTER_CUBIC, BORDER_CONSTANT, 0);
				   //cv::Mat disp = Mat( fgLeft.rows, fgLeft.cols, CV_32F );

	               Mat filteredl = Mat(srcLeft.size(),CV_8U);
                   Mat filteredr = Mat(srcRight.size(),CV_8U);
		           filteredl.setTo(Scalar(128),fgLeft);
				   filteredr.setTo(Scalar(128),fgRight);
		           
				   bilateralFilter(srcLeft,filteredl,0,20.0,2.0);
				   bilateralFilter(srcRight,filteredr,0,20.0,2.0);
		           filteredl.copyTo(frame1, fgLeft);  // Copies non-masked pixels from filtered to frame.
				   filteredr.copyTo(frame2,fgRight); 				   
				   namedWindow("frame1",0);
                   resizeWindow("frame1",300,300);
				   imshow("frame1",frame1);
				   //double minVal, maxVal;
				   //minMaxIdx(dispMat, &minVal, &maxVal);

				   double alpha = 255 * 2 / (double)dispMat.cols;
				   dispMat.convertTo(dispMatNorm, CV_8U, alpha, 0);

				   normalize(dispMatNorm, dispMatNormn, 0, 255, NORM_MINMAX);
				//showImage("disp", dispMatNorm, 0, 1);
				   sprintf_s(filename, "%s\\disp-%05d.jpg", disparity, im);
				   imwrite(string(filename), dispMatNormn);
                   //draw the depth map using reprojectImageTo3D
                   double minVal;
				   double maxVal;
				   minMaxLoc(dispMat,&minVal,&maxVal);
                   normalize(dispMatNorm, dispMatNormn, 0, 255, CV_MINMAX, CV_8U);
				   ///*namedWindow("disp",0);
       //            resizeWindow("disp",300,300);*/
				   ////showImage("disp", disp8, 0, 1);
				   char buffer[25];
                   sprintf(buffer, "min %.03f max %.03f", minVal, maxVal);
				   CvFont font;
                   cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 3, 3);//define the front size
		           Point cen(10,60);
				   putText(dispMatNormn, buffer,cen,CV_FONT_NORMAL, 2, Scalar(255),4,7); //frame is Mat class; 
				   sprintf_s(filename, "%s\\disp-%05d.jpg", disparity, im);
				   imwrite(string(filename), dispMatNormn);				   
				                  
				   //Mat filteredl = Mat(srcLeft.size(),CV_8U);
       //            Mat filteredr = Mat(srcRight.size(),CV_8U);
		     //      filteredl.setTo(Scalar(128),fgLeft);
				   //filteredr.setTo(Scalar(128),fgRight);
				   //bilateralFilter(srcLeft,filteredl,0,20.0,2.0);
				   //bilateralFilter(srcRight,filteredr,0,20.0,2.0);
		     //      filteredl.copyTo(frame1, fgLeft);  // Copies non-masked pixels from filtered to frame.
				   //filteredr.copyTo(frame2,fgRight); 	
				   //
				   //Mat disp8;                    
				   //
				   //StereoBM sbm;
       //           
				   //sbm.state->SADWindowSize = 17;
       //            sbm.state->numberOfDisparities = 288;
       //            sbm.state->preFilterSize = 5;
       //            sbm.state->preFilterCap = 61;
       //            sbm.state->minDisparity = 0;
       //            sbm.state->textureThreshold = 50;
       //            sbm.state->uniquenessRatio = 0;
       //            sbm.state->speckleWindowSize = 0;
       //            sbm.state->speckleRange = 8;
       //            sbm.state->disp12MaxDiff = 1;
				   //sbm(frame1, frame2, disp); 

       //           normalize(disp,disp8, 0, 255, NORM_MINMAX);

           
				//Mat Image3D(dispMatNormn.size(),CV_32FC3);

				  //stereoMatcher.loadCameraParameters("camera_parameter.xml");
				  //Load Matrix Q
                  //cv::FileStorage fs("camera_parame.xml", cv::FileStorage::READ);
                  ////cv::Mat Q;
  
                  //fs["Q"] >> Q;
  

				  //reprojectImageTo3D(dispMatNormn, Image3D, Q,false, CV_32F);
				  //
				  //Mat imfinal;
				  ////cv::Mat_<float> vec(4,1);
      //            for(int y=0; y<Image3D.cols; ++y) {
					 // for(int x=0; x<Image3D.rows; ++x) {
						//  imfinal.at<char>(y,x) = Image3D.at<float>(y,x);
					 // }
				  //}
				  //imshow("imfinal", imfinal);
				  //waitKey(0);
                  //for(int x=0; x<disp.cols; ++x) {
      //            vec(0)=x; vec(1)=y; vec(2)=disp.at<float>(y,x); vec(3)=1;
      //            vec = Q*vec;
      //            vec /= vec(3
				  //cv::Mat_<float> pt(4,1);
      //            cv::Vec3f &point = Image3D.at<cv::Vec3f>(y,x);
      //            point[0] = pt(0);
      //            point[1] = pt(1);
      //            point[2] = pt(2);
    
				  //}
				  //}

				   //create a file to store the lengths
				  //Mat imfinal;
				  ////cv::Mat_<float> vec(4,1);
      //            for(int y=0; y<Image3D.cols; ++y) {
					 // for(int x=0; x<Image3D.rows; ++x) {
						//  imfinal.at<char>(y,x) = Image3D.at<float>(y,x);
					 // }
				  //}
				  //imshow("imfinal", imfinal);
				  //waitKey(0);
	     //         sprintf_s(filename, "%s\\position.csv", model);
	     //         ofstream foutLen(filename, ios::out);
  
				  //ofstream MyExcelFile;
      //            MyExcelFile.open("%s\\position.csv");
 
      //           // MyExcelFile << "First Name, Last Name, Middle Initial" << endl;
      //            MyExcelFile << Image3D << endl;
      //            MyExcelFile.close();
      //            return 0;
				  //sprintf_s(filename, "%s\\length.csv", model);
	     //         ofstream foutLen(filename, ios::out);
				   bool simple(false), rgb(false), custom_c(false), normals(false),
                   shapes(false), viewports(false), interaction_customization(false);
                   /* if (pcl::console::find_argument (argc, argv, "-s") >= 0)
                   {*/
                   simple = true;
				  //#ifdef CUSTOM_REPROJECT
                  //Get the interesting parameters from Q
				  double Q03, Q13, Q23, Q32, Q33;
				  Q03 = Q.at<double>(0,3);
				  Q13 = Q.at<double>(1,3);
				  Q23 = Q.at<double>(2,3);
				  Q32 = Q.at<double>(3,2);
				  Q33 = Q.at<double>(3,3);
  
				  std::cout << "Q(0,3) = "<< Q03 <<"; Q(1,3) = "<< Q13 <<"; Q(2,3) = "<< Q23 <<"; Q(3,2) = "<< Q32 <<"; Q(3,3) = "<< Q33 <<";" << std::endl;
  
				 // #endif  
				  //stringstream output;
				  //pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
				  //#ifndef CUSTOM_REPROJECT
                  //Create matrix that will contain 3D corrdinates of each pixel
                  Mat Image3D(dispMatNormn.size(),CV_32FC3);
                  reprojectImageTo3D(dispMatNormn, Image3D, Q,false, CV_32F);  
                 
				  //Reproject image to 3D
                  std::cout << "Reprojecting image to 3D..." << std::endl;

                  //#endif 
				  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
                  double px, py, pz;
                  uchar pr, pg, pb;
 
                  for (int x = 0; x < frame1.rows; x++) {
				  uchar* rgb_ptr = frame1.ptr<uchar>(x);
				  //#ifdef CUSTOM_REPROJECT
				  uchar* disp_ptr = dispMatNormn.ptr<uchar>(x);
				  //#else
				  //double* recons_ptr = Image3D.ptr<double>(x);
				  //#endif
				  for (int j = 0; j < srcLeft.cols; j++)
				  {
      //Get 3D coordinates
				  //#ifdef CUSTOM_REPROJECT
      uchar d = disp_ptr[j];
      if ( d == 0 ) continue; //Discard bad pixels
      double pw = -1.0 * static_cast<double>(d) * Q32 + Q33; 
      px = static_cast<double>(j) + Q03;
      py = static_cast<double>(x) + Q13;
      pz = Q23;
      
      px = px/pw;
      py = py/pw;
      pz = pz/pw;

//#else
//      px = recons_ptr[3*j];
//      py = recons_ptr[3*j+1];
//      pz = recons_ptr[3*j+2];
//#endif
      
      //Get RGB info
      pb = rgb_ptr[3*j];
      pg = rgb_ptr[3*j+1];
      pr = rgb_ptr[3*j+2];

          //Insert info into point cloud structure
                  pcl::PointXYZRGB point;
                  point.x = px;
                  point.y = py;
                  point.z = pz;
                  uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
                  static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
                  point.rgb = *reinterpret_cast<float*>(&rgb);
                  point_cloud_ptr->points.push_back (point);

                  //if(p.z >= 10000) continue;  // Filter errors
                  //output << p.x << "," << p.y << "," << p.z << endl;
                  }  
				  /*ofstream outputFile("points");
                  outputFile << output.str();
                  outputFile.close();

                  cout << "saved" << endl;
                  
				  sprintf_s(filename, "%s\\disp-%05d.jpg", depth, im);
				  imwrite(string(filename), Image3D);

*/
			}
point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
point_cloud_ptr->height = 1;
//Create visualizer
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

	pcl::PolygonMesh mesh1; 
	pcl::io::loadPolygonFileOBJ ("trout.obj", mesh1); 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>); 
	pcl::fromROSMsg(mesh1.cloud, *cloud_xyz); 
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	copyPointCloud(*cloud_xyz, *cloud_xyzrgb); 
    
	pcl::concatenateFields(*point_cloud_ptr,*cloud_xyzrgb,*point_cloud_ptr);
    //viewer = View1(point_cloud_ptr);
	viewer = View2(point_cloud_ptr,cloud_xyzrgb);
    while (!viewer->wasStopped ())
     {
       viewer->spinOnce (100);
       boost::this_thread::sleep (boost::posix_time::microseconds (100000));
     }
     return 0; 

//viewer = createVisualizer( point_cloud_ptr );


  //while (!viewer->wasStopped ())
  //{
  //  viewer->spinOnce (100);
  //  boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  //}
  //return 0;
		}
	}

		//reprojectImageTo3D(dispMatNormn, OutputArray _3dImage, InputArray Q, bool handleMissingValues=false, int ddepth=-1 )
		// --------------------------------------------------------
		// downsample original L and R images for final output
		cv::Mat drawRectLeftC3 = cv::Mat::zeros(srcLeft.size(), CV_8UC3);
		cvtColor(srcLeft, drawRectLeftC3, CV_GRAY2RGB);

		cv::Mat drawRectRightC3 = cv::Mat::zeros(srcRight.size(), CV_8UC3);
		cvtColor(srcRight, drawRectRightC3, CV_GRAY2RGB);
		
		// resize the results and copy to a combined image for output to disk
		string textLeft = "Left";
		string textRight = "Right";

		cv::Mat resLeft;
		resize(drawRectLeftC3, resLeft, cv::Size(outHeight, outWidth/2), 0, 0, cv::INTER_AREA);
		cv::Mat resRight;
		resize(drawRectRightC3, resRight, cv::Size(outHeight, outWidth/2), 0, 0, cv::INTER_AREA);
		
		cv::Mat combinedImg(outHeight, outWidth, resRight.type());
		resLeft(cv::Range(0,outHeight), cv::Range(0,outWidth/2)).copyTo(combinedImg(cv::Range(0,outHeight), cv::Range(0,outWidth/2)));
		resRight(cv::Range(0,outHeight), cv::Range(0,outWidth/2)).copyTo(combinedImg(cv::Range(0,outHeight), cv::Range(outWidth/2,outWidth)));
		putText(combinedImg, textLeft, cv::Point(20, 50), CV_FONT_HERSHEY_PLAIN, 4, cvScalar(0, 255, 0), 3);
		putText(combinedImg, textRight, cv::Point(outWidth/2 + 20, 50), CV_FONT_HERSHEY_PLAIN, 4, cvScalar(0, 255, 0), 3);
		line(combinedImg, cv::Point(outWidth/2-1, 0), cv::Point(outWidth/2-1, outHeight-1), cv::Scalar::all(255), 2);

		//showImage("Combined", combinedImg, 0, 1);

		// same thing for the foreground masks
		cv::Mat resFgLeft;
		resize(fgLeft, resFgLeft, cv::Size(outHeight, outWidth/2), 0, 0, cv::INTER_AREA);
		cv::Mat resFgRight = cv::Mat::zeros(cv::Size(outHeight, outWidth/2), resFgLeft.type());
		if(fgRight.data)
			resize(fgRight, resFgRight, cv::Size(outHeight, outWidth/2), 0, 0, cv::INTER_AREA);
		
		cv::Mat combinedFgImg(outHeight, outWidth, resFgRight.type());
		resFgLeft(cv::Range(0,outHeight), cv::Range(0,outWidth/2)).copyTo(combinedFgImg(cv::Range(0,outHeight), cv::Range(0,outWidth/2)));
		resFgRight(cv::Range(0,outHeight), cv::Range(0,outWidth/2)).copyTo(combinedFgImg(cv::Range(0,outHeight), cv::Range(outWidth/2,outWidth)));
		cvtColor(combinedFgImg, combinedFgImg, CV_GRAY2BGR);
		putText(combinedFgImg, textLeft, cv::Point(20, 50), CV_FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 0), 3);
		putText(combinedFgImg, textRight, cv::Point(outWidth/2 + 20, 50), CV_FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 0), 3);
		line(combinedFgImg, cv::Point(outWidth/2-1, 0), cv::Point(outWidth/2-1, outHeight-1), cv::Scalar::all(255), 2);
		
		//sprintf_s(filename, "%s\\%05d.jpg", comparison, im);
		//imwrite(string(filename), combinedImg);

		sprintf_s(filename, "%s\\fg-%05d.jpg", model, im);
		imwrite(string(filename), combinedFgImg);
		
		cout << endl;		
	}
	
	destroyAllWindows();

	return 0;
}


int main()
{
	int ok = mainSegAndStereo();
	//pcl::PolygonMesh mesh1; 
	//pcl::io::loadPolygonFileOBJ ("trout.obj", mesh1); 
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>); 
	//pcl::fromROSMsg(mesh1.cloud, *cloud_xyz); 
	//
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZ>);
	//copyPointCloud(*cloud_xyz, *cloud_xyzrgb); 

 //   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
 //   //viewer = View1(cloud_xyzrgb);
	//viewer = View2(point_cloud_ptr,cloud_xyzrgb);
 //   while (!viewer->wasStopped ())
 //    {
 //      viewer->spinOnce (100);
 //      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
 //    }
 //    return 0; 
	 return ok;
 }
