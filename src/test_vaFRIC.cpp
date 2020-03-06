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
#define _USE_MATH_DEFINES
#define INFO

#include <math.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <regex>
using namespace std;

#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include <sophus/se3.hpp>
using namespace Sophus;

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
using namespace cv;

#include <boost/regex.hpp>
using namespace boost;

#include "VaFRIC.h"
#include "add_kinect_noise.cuh"

float K[3][3] = {9.242774047851562500e+02, 0.000000000000000000e+00, 6.400000000000000000e+02,
				 0, 7.798590698242187500e+02, 3.600000000000000000e+02,
				 0, 0, 1.00};

string getDepthFileName(const string &path_name_, int ref_image_no, int which_blur)
{
	char imageFileName[300];
	sprintf(imageFileName, "%s/scene_%02d_%04d.depth.png", path_name_.c_str(), which_blur, ref_image_no);
	string fileName(imageFileName);
	return fileName;
}

void getAllDepthImages(const string &dir_root, std::vector<string> &paths_relative_depth)
{
	// loop through dir_root;
	boost::filesystem::recursive_directory_iterator iter(dir_root), end;

	while (iter != end)
	{
		// define pattern;
		string fileName = iter->path().filename().string();
		std::regex pattern(".*IR.*png");

		// match pattern;
		if (std::regex_match(fileName.begin(), fileName.end(), pattern))
		{
			string path_abs = iter->path().string();
			string path_relative = path_abs.substr(dir_root.size(), path_abs.size() - dir_root.size());
			paths_relative_depth.push_back(path_relative);
		}

		iter++;
	}
}

void printNormalImage(const GpuMat &depth_gpu, string &name_)
{
	GpuMat vertex_gpu;
	vertex_gpu.create(depth_gpu.rows, depth_gpu.cols, CV_32FC3);
	GpuMat normal_gpu;
	normal_gpu.create(depth_gpu.rows, depth_gpu.cols, CV_32FC3);
	GpuMat normal_image_float_gpu;
	normal_image_float_gpu.create(depth_gpu.rows, depth_gpu.cols, CV_32FC3);
	GpuMat normal_image_gpu;
	normal_image_gpu.create(depth_gpu.rows, depth_gpu.cols, CV_8UC3);
	////for debug
	convertDepth2Verts(depth_gpu, &vertex_gpu, make_float2(K[0][2], K[1][2]), make_float2(K[0][0], K[1][1]));
	cudaFastNormalEstimation(vertex_gpu, &normal_gpu);
	launch_colour_from_normals((float3 *)normal_gpu.data, (float3 *)normal_image_float_gpu.data, normal_gpu.cols, normal_gpu.rows);

	normal_image_float_gpu.convertTo(normal_image_gpu, CV_8UC3);
	Mat normal_image;
	normal_image_gpu.download(normal_image);
	imwrite(name_.c_str(), normal_image);
}

int main(int argc, char **argv)
{
	// define size
	int height = 720;
	int width = 1280;

	// edit the following line to specify input folders and output folders
	string rootpath_in("../data/in/");
	string rootpath_out("../data/out/");

	// find all path to depth images
	std::vector<std::string> paths_relative_depth;
	getAllDepthImages(rootpath_in, paths_relative_depth);
	cout << "Number of IR files = " << paths_relative_depth.size() << endl;

	Mat depth_image;
	depth_image.create(height, width, CV_8UC3);
	Mat depth;
	depth.create(height, width, CV_16UC1);
	Mat depth_float;
	depth_float.create(height, width, CV_32FC1);
	Mat vertex;
	vertex.create(height, width, CV_32FC3);
	Mat normal_dist;
	normal_dist.create(height, width, CV_32FC1);

	GpuMat vertex_gpu;
	vertex_gpu.create(height, width, CV_32FC3);
	GpuMat normal_gpu;
	normal_gpu.create(height, width, CV_32FC3);
	GpuMat noisy_vertex_gpu;
	noisy_vertex_gpu.create(height, width, CV_32FC3);
	GpuMat depth_gpu;
	depth_gpu.create(height, width, CV_16UC1);
	GpuMat depth_float_gpu;
	depth_float_gpu.create(height, width, CV_32FC1);
	GpuMat depth_shifted_gpu;
	depth_shifted_gpu.create(height, width, CV_32FC1);
	GpuMat gaussian_2d_shift_gpu;
	gaussian_2d_shift_gpu.create(height, width, CV_32FC2);

	for (auto iter = paths_relative_depth.begin(); iter != paths_relative_depth.end(); iter++)
	{

		// define filepath for in and out;
		string filepath_in, filepath_out;
		filepath_in = rootpath_in + *iter;
		filepath_out = rootpath_out + *iter;

		depth = imread(filepath_in.c_str(), IMREAD_UNCHANGED);
		depth_gpu.upload(depth);
		depth_gpu.convertTo(depth_float_gpu, CV_32FC1, 1 / 65535.f);
		string ni_1 = "../data/out/ni_1.png";
		printNormalImage(depth_float_gpu, ni_1);

		//1.
		convertDepth2Verts(depth_float_gpu, &vertex_gpu, make_float2(K[0][2], K[1][2]), make_float2(K[0][0], K[1][1]));
		cudaFastNormalEstimation(vertex_gpu, &normal_gpu); // estimate surface normal
		launch_add_kinect_noise((float3 *)vertex_gpu.data,
								(float3 *)normal_gpu.data,
								(float3 *)noisy_vertex_gpu.data,
								vertex_gpu.cols,
								vertex_gpu.rows,
								720, 1.0, 0.05,
								0, 0, 0);
		//convert vertex to depth
		depth_float_gpu.setTo(0.f);
		convertVerts2Depth(&noisy_vertex_gpu, &depth_float_gpu, make_float2(K[0][2], K[1][2]), make_float2(K[0][0], K[1][1]));
		string ni_2 = "../data/out/ni_2.png";
		printNormalImage(depth_float_gpu, ni_2);

		//2.
		gaussian_shifts((float2 *)gaussian_2d_shift_gpu.data, gaussian_2d_shift_gpu.cols, gaussian_2d_shift_gpu.rows, 3.f);
		add_gaussian_shifts(depth_float_gpu, gaussian_2d_shift_gpu, &depth_shifted_gpu);
		string ni_3 = "../data/out/ni_3.png";
		printNormalImage(depth_shifted_gpu, ni_3);

		//3.
		add_depth_noise_barronCVPR2013((float *)depth_shifted_gpu.data, depth_shifted_gpu.cols, depth_shifted_gpu.rows);
		string ni_4 = "../data/out/ni_4.png";
		printNormalImage(depth_shifted_gpu, ni_4);

		//convert depth from metre to 1000 mm
		depth_shifted_gpu.convertTo(depth_gpu, CV_16UC1, 65535.f);
		depth_gpu.download(depth);

		// ready for output
		boost::filesystem::path path_parent = boost::filesystem::path(filepath_out).parent_path();
		if (!boost::filesystem::exists(path_parent))
		{
			boost::filesystem::create_directories(path_parent);
		}

		if (!imwrite(filepath_out.c_str(), depth))
		{
			cerr << "cannot write file" << endl;
		}
	}

	return 1;
}
