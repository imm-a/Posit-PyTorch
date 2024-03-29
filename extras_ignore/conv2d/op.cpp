//#ifndef _POSMUL_
//#define _POSMUL_

#include <universal/number/posit/posit.hpp>
#include <universal/utility/convert_to.hpp>
#include <universal/number/posit/table.hpp>
#include <universal/utility/convert_to.hpp>
#include <universal/internal/bitblock/bitblock.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include "genlookupmul.cpp"
#include "genlookup2.cpp"
#include <torch/script.h>
#include <ATen/ATen.h>
#include "omp.h"
using namespace std::chrono;
using namespace std;
using namespace torch::indexing;
//template<typename Real>
/*Real GetNum(const Real& a ){
return a;
}*/

//need to figure out accumulation.
double posit_mul(torch::Tensor inputmat1,torch::Tensor inputmat2,int64_t dim1, int64_t dim2,string** lookup_mul, string** lookup_add){
	const int n1 = 8;
	const int es1 = 0;
	int i; int j; float t; int k;
	sw::universal::posit<n1,es1> Arr1[dim1][dim2]; //filter
	sw::universal::posit<n1,es1> Arr2[dim1][dim2];
	double res;

	//at::Tensor tensor = at::ones({dim1, dim2}, at::kDouble);
	static string** Arr = lookup_mul;
	static string** Arr_add = lookup_add;
	sw::universal::posit<n1,es1> sum = 0.0;
	for(i=0; i<dim1; i++){
	for(j=0; j<dim2; j++){
	

	
	
	Arr1[i][j] = inputmat1[i][j].item<double>();
	Arr2[i][j] = inputmat2[i][j].item<double>();
	
	std::string x_a = info_print(Arr1[i][j]); //decode pos 1 to string
    	std::string x1_a = x_a.substr(x_a.find("raw"),x_a.find("decode"));
    	std::string x2_a = x1_a.substr(5,n1);
    	//std::cout<<"cp1"<<::endl;
    	std::string x_b = info_print(Arr2[i][j]); //decode pos 2 to string
    	std::string x1_b = x_b.substr(x_b.find("raw"),x_b.find("decode"));
    	std::string x2_b = x1_b.substr(5,n1);
    	//convert string to int index
	int idx1 = stoi(x2_a,0,2);
    	int idx2 = stoi(x2_b,0,2);
    	//find the result
    	std::string res = Arr[idx1][idx2];
    	
       //int p_=stoi(res,0,2);
   	sw::universal::posit<n1,es1> p;
    	//p.setbits(p_);
    	//p.get(); //result of each mul
    	//convert string to int index again
    	int idx3 = stoi(res,0,2);
	std::string x_s = info_print(sum); //decode pos 1 to string
    	std::string x1_s = x_s.substr(x_s.find("raw"),x_s.find("decode"));
    	std::string x2_s = x1_s.substr(5,n1);
    	int idx4 = stoi(x2_s,0,2);
    	std::string res_final = Arr_add[idx3][idx4];
    	int p_ = stoi(res_final,0,2);
    	p.setbits(p_);
    	p.get();
    	//std::cout<<p<<::endl;
    	sum = p;
	//std::cout<<Arr3[i][j]<<::endl;
	
	
	//std::cout<<"sum: "<<Arr3[i][j]<<::endl;
	//tensor[i][j] = Arr3[i][j];
	}

	}
	//delete[] Arr3;
   //torch::Tensor output = torch::from_blob(Arr3, /*sizes=*/{dim1, dim3});
	res = (double)sum;
    //std::cout<<"tens: "<<tensor<<::endl;
  //output.requires_grad_(true);
  return res;
	
}

torch::Tensor conv2d(torch::Tensor inputmat_image,torch::Tensor inputmat_filter,int64_t dim_image, int64_t dim_filter, int64_t n_mult,int64_t approx_type){
	const int n1=8;	
	const int es1=0;
	static string** Arr = create2DArrayMUL(256,256, n1, es1,approx_type); // generating lookup table
	static string** Arr_add = create2DArray(256,256, n1, es1);
	int i,j;
	double output;
	int stride = 1;
	int64_t dim_final = (dim_image - dim_filter)/stride + 1;
	//int64_t bsize = (floor((dim1-1)/n_mult)+1);
	at::Tensor output_final = at::ones({dim_final, dim_final}, at::kDouble);
	//std::cout<<bsize<<::endl;


	#pragma omp parallel for
for(i=0;i<dim_final;i++){
	torch::Tensor i1 = inputmat_image.index({Slice(i,i+dim_filter-1)});
	for(j=0;j<dim_final;j++){
	//std::cout<<"printing:"<<inputmat1.index({Slice(i*bsize,i*bsize+bsize)})<<::endl;
	//std::cout<<"mul:"<<i<<::endl;
	
 output = posit_mul(inputmat_filter,i1.index({Slice(j,j+dim_filter-1)}),dim_filter,dim_image,Arr,Arr_add);
//output = posit_mul(inputmat1.index({Slice(i*bsize,i*bsize+bsize)}),inputmat2,bsize,dim2,dim3,Arr,Arr_add);
output_final[i][j] = output;

}
}
torch::Tensor output_final_final = output_final; 
return output_final_final;
}

/*TORCH_LIBRARY(my_ops, m) {
  m.def("posit_add", posit_add);
}*/

/*int main(){


	//using Real = sw::universal::posit<8,1>; 	
	//GetNum(0.5);
	torch::Tensor tensor1 = torch::randn({3,4});
	torch::Tensor tensor2 = torch::randn({3,4});
	posit_add(tensor1,tensor2,3,4);

}*/

static auto registry = torch::RegisterOperators("my_ops::conv2d", &conv2d); //register the operator matmul
