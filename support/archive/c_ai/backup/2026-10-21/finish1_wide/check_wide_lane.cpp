#include "zc_mmvq_q4k_q6k.cuh"
#include <cstdio>
#include <random>
#include <vector>
using namespace pelias::zc;
int main(){
  std::mt19937 r(7); double worst=0; int bad=0;
  for(int t=0;t<20000;t++){
    uint8_t b[144]; for(auto&v:b) v=r()&255;
    // sane half scales: d,dmin in (0,0.1]
    uint16_t h=0x2800+(r()&0x3ff); b[0]=h&255;b[1]=h>>8; h=0x2800+(r()&0x3ff); b[2]=h&255;b[3]=h>>8;
    alignas(16) uint8_t al[144]; memcpy(al,b,144);
    std::vector<float> x(256); for(auto&v:x) v=(int(r()%2001)-1000)/1000.0f;
    double ref=0, sabs=0; for(int l=0;l<32;l++) ref+=q4k_lane(b,x.data(),l);
    double got=0; for(int c=0;c<8;c++) got+=q4k_lane_wide(al,x.data(),c);
    { // exact |terms| in double, same dequant formula -> derived bound gamma_n * sum|a x|, n=256 per path, 2 paths
      const double d0=half_to_float(ld_u16(b)), dm=half_to_float(ld_u16(b+2)); const uint8_t*sc=b+4,*qs=b+16;
      for(int g=0;g<4;g++){int s0,m0,s1,m1;scale_min_k4(2*g,sc,&s0,&m0);scale_min_k4(2*g+1,sc,&s1,&m1);
        for(int l=0;l<32;l++){int q=qs[32*g+l];
          sabs+=std::fabs((d0*s0*(q&15)-dm*m0)*x[64*g+l])+std::fabs((d0*s1*(q>>4)-dm*m1)*x[64*g+32+l]);}}}
    double d=std::fabs(got-ref); worst=std::max(worst,d);
    const double u=std::ldexp(1.0,-24), gam=512*u/(1-512*u); if(d>gam*sabs*2) bad++;
  }
  printf("worst abs diff %.3g, bad(> 2*gamma_512*sum|ax|) %d of 20000\n",worst,bad);
  return bad!=0;
}
