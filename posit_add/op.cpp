

#include <universal/number/posit/posit.hpp>
#include <universal/utility/convert_to.hpp>
#include <universal/number/posit/table.hpp>
#include <universal/utility/convert_to.hpp>
#include <universal/internal/bitblock/bitblock.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
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
//inputmat2 is simply a vector
torch::Tensor pos_add(torch::Tensor inputmat1,torch::Tensor inputmat2, int64_t dim1, int64_t dim2,string** lookup_add){
	const int n1 = 8;
	const int es1 = 3;		
		
	int i; int j; float t;
	sw::universal::posit<n1,es1> Arr1[dim1][dim2];
	sw::universal::posit<n1,es1> Arr2[dim1][dim2];
	double Arr3[dim1][dim2];
	at::Tensor tensor = at::ones({dim1, dim2}, at::kDouble);
	string** Arr = lookup_add;
	for(i=0; i<dim1; i++){
	for(j=0; j<dim2; j++){
	
	Arr1[i][j] = inputmat1[i][j].item<double>();
	Arr2[i][j] = inputmat2[0][j].item<double>();
	//std::cout<<Arr2[i][j]<<::endl;
	std::string x_a = info_print(Arr1[i][j]);
    	std::string x1_a = x_a.substr(x_a.find("raw"),x_a.find("decode"));
    	std::string x2_a = x1_a.substr(5,n1);
    	std::string x_b = info_print(Arr2[i][j]);
    	std::string x1_b = x_b.substr(x_b.find("raw"),x_b.find("decode"));
    	std::string x2_b = x1_b.substr(5,n1);
	int idx1 = stoi(x2_a,0,2);
    	int idx2 = stoi(x2_b,0,2);
    	std::string res = Arr[idx1][idx2];
       int p_=stoi(res,0,2);
   	sw::universal::posit<n1,es1> p;
    	p.setbits(p_);
    	p.get();
	Arr3[i][j] = (double)p; 
	//std::cout<<Arr3[i][j]<<::endl;
	tensor[i][j] = Arr3[i][j];
	}
	 
	}
	
  torch::Tensor output = tensor;
  return output;
  //return output.clone();
	
}
torch::Tensor posit_add(torch::Tensor inputmat1,torch::Tensor inputmat2,int64_t dim1, int64_t dim2, int64_t n_add){
	const int n1=8;	
	const int es1=3;
	//static string** Arr = create2DArrayMUL(256,256, n1, es1); // generating lookup table
	static string** Arr_add = create2DArray(256,256, n1, es1);
	int i;
	torch::Tensor output;
	//torch::Tensor output_final;
	int64_t bsize = (floor((dim1-1)/n_add)+1);
	at::Tensor output_final = at::ones({dim1, dim2}, at::kDouble);
	//std::cout<<bsize<<::endl;


	#pragma omp parallel for
for(i=0;i<n_add;i++){
	//std::cout<<"printing:"<<inputmat1.index({Slice(i*bsize,i*bsize+bsize)})<<::endl;
	//std::cout<<"add:"<<i<<::endl;
output = pos_add(inputmat1.index({Slice(i*bsize,i*bsize+bsize)}),inputmat2,bsize,dim2,Arr_add);
output_final.index_put_({Slice(i*bsize,i*bsize+bsize)},output);
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

static auto registry = torch::RegisterOperators("my_ops::posit_add", &posit_add);
