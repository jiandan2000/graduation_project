#include<iostream>
#include<fstream>
#include<string>
#include<ctime>
#include<map>
#include<set>
#include<vector>
#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<CL/cl.h>
using namespace std;
#pragma warning( disable : 4996 )   //�ر�opencl����

const int USER_SIZE = 360, ARTIST_SIZE = 6000;//�û��͸��ֵ�����
const int FACTORS = 10;                        //�������ĸ���
const float REGULARIZATION = 0.01;            //��ֹ����϶��������򻯵�����ϵ��
const float ALPHA = 1.0;                      //���Ŷ�C������ϵ��
int iterations = 15;                          //��������

vector<string>user = {}, artist{};            //���ڰ��ֵ���洢�û��͸�����Ϣ
map<pair<int, int>, int>Tempui, Tempiu;             //�洢�û��͸��ֵ��Ӧ�Ĳ��Ŵ���

const int MAPPINGNUM = 180000;
int Cui_indptr[USER_SIZE], Cui_indices[MAPPINGNUM], Cui_data[MAPPINGNUM];  //�û���Ӧ�ĸ���id�����Ӧ���Ŵ���
int artistIndex, userIndex;
int Ciu_indptr[ARTIST_SIZE],Ciu_indices[MAPPINGNUM], Ciu_data[MAPPINGNUM];    //���ֶ�Ӧ���û�id�����Ӧ���Ŵ���

float userFactors[USER_SIZE][FACTORS], artistFactors[ARTIST_SIZE][FACTORS];//��ʽ�û��������ʽ���־���

//�����Ŵ������ַ���ת��Ϊ����
int PlayNum(string num) {
	int res = 0;
	for (int i = 0; i < num.size(); i++)
		res = res * 10 + (num[i] - '0');

	return res;
}

//��ȡ�ļ�
ifstream ReadFile() {
	//ifstream infile("single_test_dataset.csv");
	ifstream infile("testdata.csv");
	//ifstream infile("code_test.csv");
	return infile;
}

//Ϊ�û��͸��ֽ�������
void CreateIndex() {
	int flag;                      //��ǵ�ǰ���ַ����������û�(0)���Ǹ���(1)
	string readrow, str;           //readrow���ڶ�ȡ��ǰ�����ݣ�str���浱ǰ��ֵ�����
	set<string>users, artists;     //�ֱ𱣴��û��͸��ֵĵ�һ��Ϣ

	//��ȡ�ļ�����
	ifstream infile = ReadFile();
	if (!infile.good()) {
		cout << "Function CreateIndex open file fail!" << endl;
		return;
	}

	//�������ݴ���
	while (getline(infile, readrow)) {
		str = "";
		flag = 0;

		for (int i = 0; i < readrow.size(); i++) {
			if (readrow[i] != ',')
				str += readrow[i];
			else {
				if (flag == 0) {       //������Ϊ0��˵����ǰΪ�û���Ϣ
					users.insert(str);
					str.clear();
					flag++;
				}
				else if (flag == 1) {  //������Ϊ1��˵����ǰΪ������Ϣ
					artists.insert(str);
					break;
				}
			}
		}
	}

	//��users��artists�е�Ԫ�ذ��ֵ��򱣴浽vertor����������
	for (string str : users)
		user.push_back(str);

	for (string str : artists)
		artist.push_back(str);

	infile.close();
}

//Ϊ�û��͸��ֵĲ��Ŵ�������ӳ��
void CreateMapping() {
	//��ȡ�ļ�
	ifstream infile = ReadFile();
	if (!infile.good()) {
		cout << "Function CreateMapping open file fail!" << endl;
		return;
	}

	//�������ݴ���
	string readrow, tuser, tartist, plays, str;//��ȡ��ǰ������;�洢��ǰ������Ϣ;�洢��ǰ�û���Ϣ;�洢���Ŵ���;�洢�ַ����������
	int flag = 0;                              //��ǵ�ǰ���ַ����������û�(0)���Ǹ���(1)
	while (getline(infile, readrow)) {
		tuser.clear();
		tartist.clear();
		str.clear();
		flag = 0;

		for (int i = 0; i < readrow.size(); i++) {
			if (readrow[i] != ',')             //�зָ���Ϊ����(,)
				str += readrow[i];
			else if (readrow[i] == ',') {
				if (flag == 0)
					tuser = str;
				else if (flag == 1)
					tartist = str;

				str.clear();
				flag++;
			}
		}

		int userIndex = find(user.begin(), user.end(), tuser) - user.begin();             //���㵱ǰ�û�������
		int artistIndex = find(artist.begin(), artist.end(), tartist) - artist.begin();   //���㵱ǰ���ֵ�����
		int counts = PlayNum(str);                                                        //���㲥�Ŵ���

		Tempui[{userIndex, artistIndex}] = counts;                                           //�����û�-���ֵĲ��Ŵ���
		Tempiu[{artistIndex, userIndex}] = counts;                                           //�������-�û��Ĳ��Ŵ���
	}

	infile.close();
}

//��map�ṹת��Ϊһά����
void ChangeMapping() {
	int flag = -1;

	for (auto it = Tempui.begin(); it != Tempui.end(); it++) {
		if ((*it).first.first != flag) {
			Cui_indptr[(*it).first.first] = artistIndex;
			flag = (*it).first.first;
		}

		Cui_indices[artistIndex] = (*it).first.second;
		Cui_data[artistIndex++] = (*it).second;
	}
	Cui_indptr[user.size()] = artistIndex;

	flag = -1;
	for (auto it = Tempiu.begin(); it != Tempiu.end(); it++) {
		if ((*it).first.first != flag) {
			Ciu_indptr[(*it).first.first] = userIndex;
			flag = (*it).first.first;
		}

		Ciu_indices[userIndex] = (*it).first.second;
		Ciu_data[userIndex++] = (*it).second;
	}
	Ciu_indptr[artist.size()] = userIndex;

	/*Tempui.clear();
	Tempiu.clear();*/
}

//��ʼ����ʽ����
void IniMatrix(float arr[][FACTORS], int row, int col) {
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			//arr[i][j] = (rand() / float(RAND_MAX));//�������0.000000-1.000000�����ȸ�����
			arr[i][j] = 0.1;//�������0.000000-1.000000�����ȸ�����
}

//opencl�Ĵ���
char* ReadKernelSourceFile(const char* filename, size_t* length) {
	FILE* file = NULL;
	size_t sourceLength;
	char* sourceString;
	int ret;
	file = fopen(filename, "rb");
	if (file == NULL) {
		printf("Cant't open %s\n", filename);
		return NULL;
	}

	fseek(file, 0, SEEK_END);
	sourceLength = ftell(file);
	fseek(file, 0, SEEK_SET);
	sourceString = (char*)malloc(sourceLength + 1);
	sourceString[0] = '\0';
	ret = fread(sourceString, sourceLength, 1, file);
	if (ret == 0) {
		printf("Cant't read source %s\n", filename);
		return NULL;
	}
	fclose(file);
	if (length != 0)
		*length = sourceLength;

	sourceString[sourceLength] = '\0';
	return sourceString;
}

//��ȡ�ں�Դ�봴��OpenCL����
cl_program CreateProgram(cl_context context, cl_device_id device, const char* filename) {
	cl_int errNum;

	cl_program program;
	size_t program_length;
	char* const source = ReadKernelSourceFile(filename, &program_length);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);

	if (program == NULL) {
		printf("Failed to create CL program from source.\n");
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS) {
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
		printf("Error in kernel: %s \n", buildLog);
		return NULL;
	}
	return program;
}

int main() {
	//Ϊ�û��͸��ֽ�������
	CreateIndex();

	//Ϊ�û���ȡ���ֵĴ�������ӳ��
	CreateMapping();

	//��mapת��Ϊһά����
	ChangeMapping();

	//�������������
	srand(time(nullptr));
	//��ʼ���û�����
	IniMatrix(userFactors, user.size(), FACTORS);
	//��ʼ�����־���
	IniMatrix(artistFactors, artist.size(), FACTORS);

	//use this to check the output of each API call
	cl_int status;

	//step1:discover and initialize the platforms
	cl_uint numplatforms = 0;
	cl_platform_id* platforms = NULL;
	//use clGetPlatformIDs() to retrieve the number of platforms
	status = clGetPlatformIDs(0, NULL, &numplatforms);

	//allocate enough space for each platform
	platforms = (cl_platform_id*)malloc(numplatforms * sizeof(cl_platform_id));
	//fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numplatforms, platforms, NULL);

	//step2:discover and initialize the devices
	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;
	//use clGetDevicesIDs() to retrieve the number of devices present
	status - clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

	//allocate enough space for each device
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	//fill in devices with clGetDevicesIDs()
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

	//step3:create a context
	cl_context context = NULL;
	//create a context using clGetContext() and associate it with the devices
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	//step4:create a command queue
	cl_command_queue cmdQueue;
	//create a command queue using clCreateCommandQueue and associate it with the device you want to execute on
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);

	//step5:create device buffers
	cl_mem buffer_userFactor;
	cl_mem buffer_artistFactor;
	cl_mem buffer_Cui_indptr;
	cl_mem buffer_Cui_indices;
	cl_mem buffer_Cui_data;
	cl_mem buffer_Ciu_indptr;
	cl_mem buffer_Ciu_indices;
	cl_mem buffer_Ciu_data;
	cl_mem buffer_YTY;
	cl_mem buffer_XTX;

	//step6:user clCreateBuffer() to create a buffer object that will contain the data from the host array 
	buffer_userFactor = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, user.size() * FACTORS*sizeof(float), NULL, &status);
	buffer_artistFactor = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, artist.size() * FACTORS * sizeof(float), NULL, &status);
	buffer_Cui_indptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, USER_SIZE*sizeof(int), NULL, &status);
	buffer_Cui_indices = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MAPPINGNUM*sizeof(int), NULL, &status);
	buffer_Cui_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MAPPINGNUM*sizeof(int), NULL, &status);
	buffer_Ciu_indptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ARTIST_SIZE*sizeof(int), NULL, &status);
	buffer_Ciu_indices = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MAPPINGNUM*sizeof(int), NULL, &status);
	buffer_Ciu_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MAPPINGNUM*sizeof(int), NULL, &status);

	//use clEnqueueWriteBuffer() to write input array 
	status = clEnqueueWriteBuffer(cmdQueue, buffer_userFactor, CL_TRUE, 0, user.size() * FACTORS*sizeof(float), userFactors, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_artistFactor, CL_TRUE, 0, artist.size() * FACTORS * sizeof(float), artistFactors, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_Cui_indptr, CL_TRUE, 0, USER_SIZE * sizeof(int), Cui_indptr, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_Cui_indices, CL_TRUE, 0, ARTIST_SIZE * sizeof(int), Cui_indices, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_Cui_data, CL_TRUE, 0, MAPPINGNUM * sizeof(int), Cui_data, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_Ciu_indptr, CL_TRUE, 0, ARTIST_SIZE * sizeof(int), Ciu_indptr, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_Ciu_indices, CL_TRUE, 0, MAPPINGNUM * sizeof(int), Ciu_indices, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, buffer_Ciu_data, CL_TRUE, 0, MAPPINGNUM*sizeof(int), Ciu_data, 0, NULL, NULL);

	//step7:create and compile the program
	cl_program program = CreateProgram(context, *devices, "als_cg.cl");
	//Build(compile) the program for the devices with clBuildProgram()
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	//step8:craete the kernel
	cl_kernel kernel = NULL;
	//use clCreateKernel() to create a kernel from the vector addition (named "vecadd")
	kernel = clCreateKernel(program, "LS_CG", &status);

	//step10:configure the work-item struture
	//define an index space (global work size) of work items for execution .
	const size_t globalWorkSize[1] = { 1024*FACTORS };
	const size_t localWorkSize[1] = { FACTORS };

	//step9:set the kernel arguments
	//associate the input and output buffers with the kernel using clSetKernelArg()
	int userSize = user.size(), artistSize = artist.size(), cg_step = 3;
	float YTY[FACTORS][FACTORS] = {}, XTX[FACTORS][FACTORS] = {};
	status = clSetKernelArg(kernel, 0, sizeof(int), &FACTORS);
	while (iterations--) {
		memset(YTY, 0, sizeof(YTY));
		for (int i = 0; i < FACTORS; i++)          //����YYT
			for (int j = 0; j < FACTORS; j++)
				for (int k = 0; k < artistSize; k++)
					YTY[i][j] += artistFactors[k][i] * artistFactors[k][j];
		buffer_YTY = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FACTORS * FACTORS * sizeof(float), NULL, &status);
		status = clEnqueueWriteBuffer(cmdQueue, buffer_YTY, CL_TRUE, 0, FACTORS * FACTORS * sizeof(float), YTY, 0, NULL, NULL);

		status = clSetKernelArg(kernel, 1, sizeof(int), &userSize);
		status = clSetKernelArg(kernel, 2, sizeof(int), &artistSize);
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_userFactor);
		status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_artistFactor);
		status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_YTY);
		status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &buffer_Cui_indptr);
		status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &buffer_Cui_indices);
		status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &buffer_Cui_data);
		status = clSetKernelArg(kernel, 9, sizeof(int), &cg_step);

		//step11:EnQueue the kernel for execution
		//execute the kernel by using clEnqueueNDRangKernel().
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

		//step12:read the output buffer back to the host
		//use clEnqueueReadBuffer() to read the OpenCl output buffer(bufferC)
		clEnqueueReadBuffer(cmdQueue, buffer_userFactor, CL_TRUE, 0, user.size() * FACTORS*sizeof(float), userFactors, 0, NULL, NULL);
		
		
		memset(XTX, 0, sizeof(XTX));
		for (int i = 0; i < FACTORS; i++)          //����XXT
			for (int j = 0; j < FACTORS; j++)
				for (int k = 0; k < userSize; k++)
					XTX[i][j] += userFactors[k][i] * userFactors[k][j];
		buffer_XTX = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FACTORS * FACTORS * sizeof(float), NULL, &status);
		status = clEnqueueWriteBuffer(cmdQueue, buffer_XTX, CL_TRUE, 0, FACTORS * FACTORS * sizeof(float), XTX, 0, NULL, NULL);

		status = clSetKernelArg(kernel, 1, sizeof(int), &artistSize);
		status = clSetKernelArg(kernel, 2, sizeof(int), &userSize);
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_artistFactor);
		status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_userFactor);
		status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_XTX);
		status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &buffer_Ciu_indptr);
		status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &buffer_Ciu_indices);
		status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &buffer_Ciu_data);
		status = clSetKernelArg(kernel, 9, sizeof(int), &cg_step);

		//step11:EnQueue the kernel for execution
		//execute the kernel by using clEnqueueNDRangKernel().
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

		//step12:read the output buffer back to the host
		//use clEnqueueReadBuffer() to read the OpenCl output buffer(bufferC)
		clEnqueueReadBuffer(cmdQueue, buffer_artistFactor, CL_TRUE, 0, artist.size() * FACTORS*sizeof(float), artistFactors, 0, NULL, NULL);
	}

	//Free OpenCl resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(buffer_userFactor);
	clReleaseMemObject(buffer_artistFactor);
	clReleaseMemObject(buffer_Cui_indptr);
	clReleaseMemObject(buffer_Cui_indices);
	clReleaseMemObject(buffer_Cui_data);
	clReleaseMemObject(buffer_Ciu_indptr);
	clReleaseMemObject(buffer_Ciu_indices);
	clReleaseMemObject(buffer_Ciu_data);
	clReleaseContext(context);

	//Free host resources
	free(platforms);
	free(devices);

	//��ӡ�������ѹ��������˵�Ԥ����
	for (int i = 0; i < user.size(); i++) {
		for (int j = 0; j < artist.size(); j++) {
			float sum = 0;
			for (int k = 0; k < FACTORS; k++)
				sum += userFactors[i][k] * artistFactors[j][k];
			printf("%9.6f ", sum);
		}
		cout << endl;
	}

	return 0;
}