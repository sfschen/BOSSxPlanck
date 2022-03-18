# BOSSxPlanck/rsd_data

Directory housing the redshift-space data that we use.

Data are taken from publically available BOSS collaboration release. The Fourier-space data are taken from

https://fbeutler.github.io/hub/deconv_paper.html

while the real-space data are those associated with

https://arxiv.org/abs/1610.03506

(we thank Mariana Vargas for providing the specific data vectors and moss measurements).

The mock catalogs themselves are described in 

https://arxiv.org/abs/1509.06400.


Specifically we use the data vectors in 

pk/pk_(NGC/SGC)_z(1/3).dat

which contain kcen / P0 / P2 for each sample over 40 bins, and

xi/(z1/z3).xi

for the post-reconstruction correlation functions over 36 radial bins up to from 2.5 - 177.5 Mpc/h.

The covariances for each redshift-bin are in

covariances/cov_joint_NGCSGCZi_(z1/z3).dat

and follow the ordering NGC P0, P2, SGC P0, P2, Xi0, Xi2.




