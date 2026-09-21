#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
extern "C" void dgesvd_(const char*,const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*);
namespace glasshelix::interrogation {
struct local_sensitivity_report { std::vector<double> singular_values; double threshold=0; std::size_t rank=0; double residual=0; std::vector<double> right_nullspace; std::vector<double> parameter_scales; std::string local_domain; };
inline local_sensitivity_report rank_revealing_svd(std::span<const double> row_major,std::size_t rows,std::size_t columns,std::span<const double> scales,double threshold,std::string domain) {
 if(!rows||!columns||row_major.size()!=rows*columns||scales.size()!=columns||!(threshold>0)||domain.empty()) throw std::invalid_argument("invalid local sensitivity diagnostic");
 std::vector<double> a(rows*columns); for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<columns;++c){if(!(scales[c]>0)||!std::isfinite(row_major[r*columns+c]))throw std::invalid_argument("invalid sensitivity scale");a[c*rows+r]=row_major[r*columns+c]/scales[c];}
 int m=rows,n=columns,lda=rows,ldu=1,ldvt=columns,info=0,lwork=-1; double work=0;std::vector<double>s(std::min(rows,columns)),u(1),vt(columns*columns);char job='N',jobv='A';dgesvd_(&job,&jobv,&m,&n,a.data(),&lda,s.data(),u.data(),&ldu,vt.data(),&ldvt,&work,&lwork,&info);if(info)throw std::runtime_error("LAPACK SVD workspace query failed");lwork=static_cast<int>(work);std::vector<double>w(lwork);dgesvd_(&job,&jobv,&m,&n,a.data(),&lda,s.data(),u.data(),&ldu,vt.data(),&ldvt,w.data(),&lwork,&info);if(info)throw std::runtime_error("LAPACK SVD failed");std::size_t rank=0;for(auto x:s)if(x>threshold)++rank;std::vector<double> nullspace;for(std::size_t i=rank;i<columns;++i)nullspace.insert(nullspace.end(),vt.begin()+i*columns,vt.begin()+(i+1)*columns);return {std::move(s),threshold,rank,0.,std::move(nullspace),{scales.begin(),scales.end()},std::move(domain)};
}
}
