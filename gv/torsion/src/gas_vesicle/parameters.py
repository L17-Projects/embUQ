import numpy as np
#GV parameters / nondimensional units
#radius = 2.0		#GV radius # e.g. 50 nm 85
#height = 14.28     #14.28	#GV height # e.g. 250 nm 500 #11.76
#k0 = 40		    #number of layers # find in literature # even number is preferrable
den_fac = 1.2   #increase the density of points around GV # value den_fac = 1.0 is generally good # for small radii increase den_fac to 2 or more


frac =      0.15	#cone height/height of the cylinder # find in literature
cap =       0.0	#capped cone # find in literature # try cap = 0.1
tip =       0		#0 = circular, 1 = spiral # reality = spiral / nicer mesh = circular #2 = sphere
staggered = True #True    #staggered circles ... True probably better

#ratio =	    1 / np.sqrt(1+radius**2/frac**2/height**2)   #dz_cone/dz_middle


#CGAL parameters
#adjust so that you get a properly triangulated object
#sometimes sharp edges are a problem #usually one needs to increase the "mesher" parameter

#sphere tip = 2
#tip = 2
#npts = 10

#mesher choice
mesher =    0	    #0 = scale space, 1 = alpha - usually better for the capped ends, 2 = advancing front with RANSAC edge detection

#scale_space_advancing_front (using jet smoother)
scale =     10		#scale iterations
af1 =       40.0 	#maximum facet length

#scale_space_alpha (using weighted pca smoother)
neigh =     6  		#neighbors	is the number of neighbors a point's neighborhood should contain on average
samples =   10000 	#samples	is the number of points sampled to estimate the neighborhood radius.
scalea =    1		#scale iterations

#afm2
p1 = 	    0.05    #epsilon
p2 = 	    0.01    #cluster_epsilon
p3 = 	    0.90150 #probability
p4 =        500     #min_points
p5 =        0.70    #threshold


