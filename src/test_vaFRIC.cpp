/* Copyright (c) 2013 Ankur Handa and Shuda Li
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/
#define  _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <stdio.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
using namespace Eigen;
using namespace Sophus;
#include <opencv2/opencv.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
using namespace cv;

#define INFO
#include "VaFRIC.h"


#include "add_kinect_noise.cuh"

using namespace std;
using namespace cv;

// float K[3][3] = { 481.20,   0,  319.50,
// 					0,  -480.00,  239.50,
// 					0,        0,    1.00 };
float K[3][3] = { 4.621387023925781250e+02,   0.000000000000000000e+00,  3.200000000000000000e+02,
				0,  5.199060668945312500e+02,  2.400000000000000000e+02,
				0,        0,    1.00 };

// string getImageFileName( const string& path_name_, int ref_image_no, int which_blur){
// 	char imageFileName[300];
// 	sprintf(imageFileName,"%s/scene_%02d_%04d.png",path_name_.c_str(),which_blur,ref_image_no);
// 	string fileName(imageFileName);
// 	return fileName;
// }
string getDepthFileName( const string& path_name_, int ref_image_no, int which_blur){
	char imageFileName[300];
	sprintf(imageFileName,"%s/scene_%02d_%04d.depth.png",path_name_.c_str(),which_blur,ref_image_no);
	string fileName(imageFileName);
	return fileName;
}
int init(const string& filebasename){
	int depthfilecount = 0;
	boost::filesystem::directory_iterator itrEnd;
	for ( boost::filesystem::directory_iterator itrDir( filebasename );   itrDir != itrEnd;  ++itrDir )	{
		if( itrDir->path().extension() == ".depth" ) {
			depthfilecount++;
		}
	}
	return depthfilecount;
}

void printNormalImage(const GpuMat& depth_gpu, string& name_){
	GpuMat vertex_gpu; vertex_gpu.create( depth_gpu.rows, depth_gpu.cols,  CV_32FC3 );
	GpuMat normal_gpu; normal_gpu.create( depth_gpu.rows, depth_gpu.cols, CV_32FC3 );
	GpuMat normal_image_float_gpu; normal_image_float_gpu.create( depth_gpu.rows, depth_gpu.cols, CV_32FC3 );
	GpuMat normal_image_gpu; normal_image_gpu.create( depth_gpu.rows, depth_gpu.cols, CV_8UC3 );
	////for debug
	convertDepth2Verts( depth_gpu, &vertex_gpu, make_float2(K[0][2],K[1][2]), make_float2(K[0][0],K[1][1]) );
	cudaFastNormalEstimation( vertex_gpu, &normal_gpu );
	launch_colour_from_normals( (float3*)normal_gpu.data , (float3*)normal_image_float_gpu.data, normal_gpu.cols, normal_gpu.rows );

	normal_image_float_gpu.convertTo(normal_image_gpu,CV_8UC3);
	Mat normal_image; normal_image_gpu.download(normal_image);
	imwrite(name_.c_str(),normal_image);
}

//boost::shared_ptr<ifstream> _pInputFile;
int idx;
//vector<SE3Group<double>> _v_T_wc;

int main(int argc, char** argv)
{
	// edit the following line to specify input folders and output folders
	string input_path_name ("../data/in/");
	string output_path_name("../data/out/");
	const int nTotal = init(input_path_name);
    cout<<"Number of text files = " << nTotal << endl;

	Mat depth_float; depth_float.create(480,640,CV_32FC1);
	Mat depth; depth.create(480,640,CV_16UC1);
	Mat vertex; vertex.create(480,640,CV_32FC3);
	Mat normal_dist; normal_dist.create(480,640,CV_32FC1);

	GpuMat vertex_gpu; vertex_gpu.create(480,640,CV_32FC3);
	GpuMat normal_gpu; normal_gpu.create(480,640,CV_32FC3);
	GpuMat noisy_vertex_gpu; noisy_vertex_gpu.create(480,640,CV_32FC3);
	GpuMat depth_gpu; depth_gpu.create(480,640,CV_16UC1);
	GpuMat depth_float_gpu; depth_float_gpu.create(480,640,CV_32FC1);
	GpuMat depth_shifted_gpu; depth_shifted_gpu.create(480,640,CV_32FC1);
	GpuMat gaussian_2d_shift_gpu; gaussian_2d_shift_gpu.create(480,640,CV_32FC2);
	GpuMat normal_image_gpu_float; normal_image_gpu_float.create(480,640,CV_32FC3);
	GpuMat normal_image_gpu; normal_image_gpu.create(480,640,CV_8UC3);

	float3 sigma_c = make_float3(0.0045f, 0.0038f, 0.005f);
	float3 sigma_s = make_float3(0.0104f, 0.0066f, 0.0106f);

	Mat noisy_image; noisy_image.create( 480, 640, CV_8UC3 );
	Mat normal_image; 
	
	for (int img_no=0; img_no< nTotal; img_no++)
	{

		string depName = getDepthFileName( input_path_name, img_no, 0 );
		depth = imread( depName.c_str(), IMREAD_UNCHANGED );
		depth_gpu.upload( depth );
		depth_gpu.convertTo( depth_float_gpu, CV_32FC1, 1/1000.f );
		string ni_1 = "../data/out/ni_1.png";
		printNormalImage(depth_float_gpu, ni_1);

		//1. 
		convertDepth2Verts( depth_float_gpu, &vertex_gpu, make_float2(K[0][2],K[1][2]), make_float2(K[0][0],K[1][1]) );
		cudaFastNormalEstimation( vertex_gpu, &normal_gpu ); // estimate surface normal
		launch_add_kinect_noise( (float3*)vertex_gpu.data,
								 (float3*)normal_gpu.data,
								 (float3*)noisy_vertex_gpu.data,
								 vertex_gpu.cols,
								 vertex_gpu.rows,
								 480, 0.8, 0.035,
								 0, 0, 0 );
		//convert vertex to depth
		depth_float_gpu.setTo(0.f);
		convertVerts2Depth( &noisy_vertex_gpu, &depth_float_gpu, make_float2(K[0][2],K[1][2]), make_float2(K[0][0],K[1][1]) );
		string ni_2 = "../data/out/ni_2.png";
		printNormalImage(depth_float_gpu, ni_2);

		//2. 
		gaussian_shifts( (float2*)gaussian_2d_shift_gpu.data, gaussian_2d_shift_gpu.cols, gaussian_2d_shift_gpu.rows, 3.f );
		add_gaussian_shifts( depth_float_gpu, gaussian_2d_shift_gpu, &depth_shifted_gpu );
		string ni_3 = "../data/out/ni_3.png";
		printNormalImage( depth_shifted_gpu, ni_3);
		
		//3.
		add_depth_noise_barronCVPR2013( (float*)depth_shifted_gpu.data, depth_shifted_gpu.cols, depth_shifted_gpu.rows );
		string ni_4 = "../data/out/ni_4.png";
		printNormalImage( depth_shifted_gpu, ni_4);

		//convert depth from metre to 1000 mm 
		depth_shifted_gpu.convertTo( depth_gpu, CV_16UC1, 1000 );
		depth_gpu.download(depth);

		string out_depName = getDepthFileName( output_path_name, img_no, 0 );
		imwrite( out_depName.c_str(), depth );
		cout << img_no << " ";
	}

	return 1;
}
