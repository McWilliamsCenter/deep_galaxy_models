
# based on equation 3 of Lotz, Primack, & Madau (2004)
Gini_statistic = function(img)
{
  w = which(img!=0,arr.ind=T)
  v = sort(as.vector(img[w]))
  n = length(v)
  coeff = 2*(1:n)-n-1
  gini = sum(coeff*v)/mean(v)/n/(n-1)
  return(gini)
}

# based on equations 7 and 8 of Lotz, Primack, & Madau (2004)
# this can probably be made faster/more efficient via use of "apply"
M20_statistic = function(img)
{
  # "The center is computed by finding x_c, y_c such that M_tot is minimized."
  nx = dim(img)[1]
  ny = dim(img)[2]
  xi = 1:nx
  yi = 1:ny
  xcnum = 0
  xcden = 0
  for ( ii in 1:nx ) { 
    xcnum = xcnum+sum(img[ii,]*xi[ii])
    xcden = xcden+sum(img[ii,])
  }
  xc = xcnum/xcden
  ycnum = 0
  ycden = 0
  for ( ii in 1:ny ) { 
    ycnum = ycnum+sum(img[,ii]*yi[ii])
    ycden = ycden+sum(img[,ii])
  }
  yc = ycnum/ycden

  mtot = 0
  for ( ii in 1:nx ) { 
    for ( jj in 1:ny ) { 
      mtot = mtot+img[ii,jj]*((ii-xc)^2+(jj-yc)^2)
    }
  }

  w = which(img!=0,arr.ind=T)
  v = sort(as.vector(img[w]),decreasing=T)
  n = length(v)

  ftot = sum(v)
  fsum = 0
  ii = 1
  m20 = 0
  while ( fsum < 0.2*ftot ) {
    w = which(img==v[ii],arr.ind=T)
    for ( jj in 1:(length(w)/2) ) {
      m20 = m20 + img[w[jj,1],w[jj,2]]*((w[jj,1]-xc)^2+(w[jj,2]-yc)^2)
    }
    fsum = fsum + img[w[jj,1],w[jj,2]]
    ii = ii+1
  }

  m20 = log10(m20/mtot)
  return(m20)
}


