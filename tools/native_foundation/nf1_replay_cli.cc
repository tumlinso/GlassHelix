#include <GlassHelix/artifacts/nf1_replay.hh>
#include <iostream>
int main(int argc,char**argv){if(argc!=2)return 2;glasshelix::artifacts::nf1_replay_record r{};std::string e;if(!glasshelix::artifacts::read_nf1_replay(argv[1],&r,&e)){std::cerr<<e<<'\n';return 1;}float x{};if(!glasshelix::artifacts::replay_logical_sum(r,&x,&e)){std::cerr<<e<<'\n';return 1;}std::cout<<x<<'\n';}
