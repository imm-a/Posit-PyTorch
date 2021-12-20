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
#include <chrono>
#include "genlookupmul.cpp"
#include "genlookup2.cpp"
#include <torch/script.h>
#include <ATen/ATen.h>
#include "omp.h"
using namespace std::chrono;
using namespace std;

torch::Tensor distrib(torch::Tensor inputmat, int64_t dim1, int64_t dim2){
	const int n1=8;	
	const int es1=2;	
	int i; int j; float t; int k;
	sw::universal::posit<n1,es1> Arr1[dim1][dim2];
	
	at::Tensor tensor = at::ones({dim1, dim2}, at::kDouble);
	#pragma omp parallel for
	for(i=0; i<dim1; i++){
	for(j=0; j<dim2; j++){
	Arr1[i][j] = inputmat[i][j].item<double>();
	tensor[i][j] = (double) Arr1[i][j];
	}
	
	}
	
   
   torch::Tensor output = tensor;
    
  return output;
	
}



static auto registry = torch::RegisterOperators("my_ops::distrib", &distrib);
