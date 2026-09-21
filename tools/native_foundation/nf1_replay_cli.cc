#include <GlassHelix/artifacts/nf1_replay.hh>
#include <iostream>
int main(int argc,char**argv){if(argc!=4)return 2;glasshelix::artifacts::nf1_replay_record r{};std::string e;if(!glasshelix::artifacts::read_nf1_replay(argv[1],&r,&e)||!glasshelix::artifacts::replay_matches_runtime(r,argv[2],argv[3],&e)){std::cerr<<e<<'\n';return 1;}std::vector<float>x;if(!glasshelix::artifacts::replay_defined_operation(r,&x,&e)){std::cerr<<e<<'\n';return 1;}std::cout<<x[0]<<'\n';}
