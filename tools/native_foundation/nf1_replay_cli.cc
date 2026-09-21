#include <GlassHelix/artifacts/nf1_replay.hh>
#include <iostream>
#include <string_view>
int main(int argc,char**argv){using namespace glasshelix::artifacts;if(argc==2&&std::string_view(argv[1])=="--identity"){const auto id=runtime_identity();std::cout<<id.glasshelix_source<<' '<<id.cellerator_source<<'\n';return 0;}if(argc!=2)return 2;nf1_replay_record r{};std::string e;if(!read_nf1_replay(argv[1],&r,&e)||!replay_matches_runtime(r,&e)){std::cerr<<e<<'\n';return 1;}std::vector<float>x;if(!replay_defined_operation(r,&x,&e)){std::cerr<<e<<'\n';return 1;}std::cout<<x[0]<<'\n';}
