#
# creates segmentation map (i.e., mask) around galaxy
#
library(SDMTools)

segmap = function(img,eta=0.2,thrlev=10)
{
  nx = dim(img)[1]
  ny = dim(img)[2]
  xcen = as.integer(nx/2)
  ycen = as.integer(ny/2)
  xcenlo = as.integer(nx/2)-5
  xcenhi = as.integer(nx/2)+5
  ycenlo = as.integer(ny/2)-5
  ycenhi = as.integer(ny/2)+5
  simg = sort(as.vector(img))
  npix = length(simg)
  level = seq(0.99,0,by=-0.005)
  nlevel = length(level)
  mu = -9
  for ( ii in 1:nlevel ) {
    c = ConnCompLabel(img>simg[as.integer(level[ii]*npix)])
    if ( is.null(dim(c)) == T ) return(NULL)
    if ( c[xcen,ycen] == 0 ) next;
    w = which(c==c[xcen,ycen],arr.ind=T)
    if ( mu > 0 ) {
      dnw = length(w)/2-nw
      if ( dnw < 16 ) next;
      dmu = (sum(img[w])-mu*nw)/dnw
      nw = length(w)/2
      if ( dnw > 1.1*nx*ny/200 && ii > thrlev ) {
        c = ConnCompLabel(img>simg[as.integer(level[ii-1]*npix)])
        dmu = 0
      }
      if ( dmu/(mu+dmu) < eta ) {
        allc = c
        w = which(allc>0,arr.ind=T)
        allc[w] = 1
        w = which(c!=c[xcen,ycen],arr.ind=T)
        c[w] = 0
        c = c/max(c)
        istop = 0
        while ( istop == 0 ) {
          fillc = array(0,c(nx,ny))
          for ( jj in 2:(nx-1) ) {
            for ( kk in 2:(ny-1) ) {
              if ( c[jj,kk] == 0 ) {
                if ( sum(c[(jj-1):(jj+1),(kk-1):(kk+1)]) > 4 ) {
                  fillc[jj,kk] = 1
                }
              } 
            }
          }   
          if ( sum(fillc) == 0 ) istop = 1
          c = c+fillc
        }
        return(list(c=c,allc=allc))
      }
    }
    mu = mean(img[w])
    nw = length(w)/2 
  }
  allc = c
  w = which(allc>0,arr.ind=T)
  allc[w] = 1
  w = which(c!=c[xcen,ycen],arr.ind=T)
  c[w] = 0
  c = c/max(c)
  return(list(c=c,allc=allc))
}

