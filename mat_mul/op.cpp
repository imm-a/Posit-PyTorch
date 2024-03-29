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


torch::Tensor posit_mul(torch::Tensor inputmat1,torch::Tensor inputmat2,int64_t dim1, int64_t dim2,int64_t dim3,string** lookup_mul, string** lookup_add){
	const int n1 = 8;
	const int es1 = 2;
	assert(("n1 must be 6,7 or 8", n1>=6 && n1<=8));
	assert(("Check es1 value", es1<(n1-3) && es1>=0));
	int i; int j; float t; int k;
	sw::universal::posit<n1,es1> Arr1[dim1][dim2];
	sw::universal::posit<n1,es1> Arr2[dim2][dim3];
	double Arr3[dim1][dim3];
	
	at::Tensor tensor = at::ones({dim1, dim3}, at::kDouble);
	static string** Arr = lookup_mul;
	static string** Arr_add = lookup_add;

	for(i=0; i<dim1; i++){
	for(j=0; j<dim3; j++){
	sw::universal::posit<n1,es1> sum = 0.0;

	for(k=0; k<dim2; k++){
	
	Arr1[i][k] = inputmat1[i][k].item<double>();
	Arr2[k][j] = inputmat2[k][j].item<double>();
	
	std::string x_a = info_print(Arr1[i][k]); //decode pos 1 to string
    	std::string x1_a = x_a.substr(x_a.find("raw"),x_a.find("decode"));
    	std::string x2_a = x1_a.substr(5,n1);
    	//std::cout<<"cp1"<<::endl;
    	std::string x_b = info_print(Arr2[k][j]); //decode pos 2 to string
    	std::string x1_b = x_b.substr(x_b.find("raw"),x_b.find("decode"));
    	std::string x2_b = x1_b.substr(5,n1);
    	//convert string to int index
	int idx1 = stoi(x2_a,0,2);
    	int idx2 = stoi(x2_b,0,2);
    	//find the result
    	std::string res = Arr[idx1][idx2];
    	
       //int p_=stoi(res,0,2);
   	sw::universal::posit<n1,es1> p;
    	
    	//convert string to int index again
    	int idx3 = stoi(res,0,2);
	std::string x_s = info_print(sum); //decode pos to string
    	std::string x1_s = x_s.substr(x_s.find("raw"),x_s.find("decode"));
    	std::string x2_s = x1_s.substr(5,n1);
    	int idx4 = stoi(x2_s,0,2);
    	std::string res_final = Arr_add[idx3][idx4];
    	int p_ = stoi(res_final,0,2);
    	p.setbits(p_);
    	p.get();
    	
    	sum = p;

	}
	Arr3[i][j] = (double)sum;
	
	tensor[i][j] = Arr3[i][j];
	}

	}
	
   torch::Tensor output = tensor;
    
  return output;
	
}

torch::Tensor mat_mul(torch::Tensor inputmat1,torch::Tensor inputmat2,int64_t dim1, int64_t dim2,int64_t dim3, int64_t n_mult,int64_t approx_type){
	const int n1=8;	
	const int es1=2;
	assert(("n1 must be 6,7, or 8", n1>=6 && n1<=8));
	assert(("Check es1 value", es1<(n1-3) && es1>=0));
	static string** Arr = create2DArrayMUL(int(pow(2,n1)),int(pow(2,n1)), n1, es1,approx_type); // generating lookup table
	static string** Arr_add = create2DArray(int(pow(2,n1)),int(pow(2,n1)), n1, es1);
	int i;
	torch::Tensor output;
	
	int64_t bsize = (floor((dim1-1)/n_mult)+1);
	at::Tensor output_final = at::ones({dim1, dim3}, at::kDouble);
	

	#pragma omp parallel for
for(i=0;i<n_mult;i++){
	
output = posit_mul(inputmat1.index({Slice(i*bsize,i*bsize+bsize)}),inputmat2,bsize,dim2,dim3,Arr,Arr_add);
output_final.index_put_({Slice(i*bsize,i*bsize+bsize)},output);

}
torch::Tensor output_final_final = output_final; 
return output_final_final;
}



static auto registry = torch::RegisterOperators("my_ops::mat_mul", &mat_mul); //register the operator matmul
