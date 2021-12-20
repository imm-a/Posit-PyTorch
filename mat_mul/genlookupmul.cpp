#ifndef _GEN_LOOKUP_81m_
#define _GEN_LOOKUP_81m_

#include <universal/number/posit/posit.hpp>
#include <universal/utility/convert_to.hpp>
#include <universal/number/posit/table.hpp>
#include <universal/internal/bitblock/bitblock.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdio>
using namespace std::chrono;
using namespace std;
template<typename Real>


Real GetNum(const Real& a) {
	return a;
}



    // Returns a pointer to a newly created 2d array the array2D has size [height x width]

std::string** create2DArrayMUL(int height, int width, int nval, int esval, int approx_type){
	//using Real = sw::universal::posit<8,1>; 
	
	std::fstream newfile;
    std::fstream newfile1;
    string** Arr = 0;
     Arr = new string*[height];
    
     for (int h = 0; h < height; h++)
     {
         Arr[h] = new string[width];
    
         for (int w = 0; w < width; w++)
          {
                  Arr[h][w] = "x";
           }
      }
    std::string approx = "Accurate";
    if(approx_type==1){
        approx = "Approximate_1";
    }
    if(approx_type==2){
approx = "Approximate_2";}
   // newfile.open("/home/amritha/Project/smallPosit/PositMultiplier_V/"+approx+"/"+std::to_string(nval)+"_"+std::to_string(esval)+"/testcases"+std::to_string(nval)+""+std::to_string(esval)+".txt", ios::in);
 //	newfile1.open("/home/amritha/Project/smallPosit/PositMultiplier_V/"+approx+"/"+std::to_string(nval)+"_"+std::to_string(esval)+"/compare"+std::to_string(nval)+""+std::to_string(esval)+".txt", ios::in);
	
    newfile.open("lookup/PositMultiplier_V/"+approx+"/"+std::to_string(nval)+"_"+std::to_string(esval)+"/testcases"+std::to_string(nval)+""+std::to_string(esval)+".txt", ios::in);
    newfile1.open("lookup/PositMultiplier_V/"+approx+"/"+std::to_string(nval)+"_"+std::to_string(esval)+"/compare"+std::to_string(nval)+""+std::to_string(esval)+".txt", ios::in);
    
    if(newfile.is_open() && newfile1.is_open()){
		std::string tp;
		std::string tp1;
		while(getline(newfile,tp) && getline(newfile1,tp1)){
			
		 int idx1; int idx2;
		 idx1 = stoi(tp.substr(0,nval),0,2);
		 idx2 = stoi(tp.substr(nval,nval),0,2);
		 Arr[idx1][idx2] = tp1;
		 
		}
	}
	newfile.close();
	newfile1.close();
    std::cout<<"built lookup table"<<endl;
      return Arr;
      
}
 #endif   
