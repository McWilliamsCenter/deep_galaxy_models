
dist_circle = function(imagedim,xc,yc)
{
  nx = imagedim[1]
  ny = imagedim[2]
  aimage = array(-9,c(nx,ny))
  for ( ii in 1:nx ) {
    dx = ii-xc
    for ( jj in 1:ny ) {
      dy = jj-yc
      aimage[ii,jj] = sqrt(dx^2+dy^2)
    }
  }
  return(aimage)
}

pet_rad_cir2 = function(image,xc,yc,eta=0.2)
{
  nx    = dim(image)[1]
  ny    = dim(image)[2]
  n     = as.integer(ceiling(sqrt((nx/2)^2+(ny/2)^2)))
  cir   = dist_circle(dim(image),xc,yc)
  mu    = rep(-9,n)
  R_pet = rep(n,n)

  sum_int = 0

  # compute curve of growth
  for ( r in 1:(n-1) ) {
    if ( r == 1 ) {
      ann = which(cir<r+0.5,arr.ind=T)
    } else {
      ann = which(cir>=r-0.5&cir<r+0.5,arr.ind=T)
    }
    if ( length(ann) == 0 ) break;
    mu[r] = sum(image[ann])/(length(image[ann])/2)
    if ( mu[r] < 0 ) mu[r]=0 
    tot = which(cir<r+0.5,arr.ind=T)
    avg_mu = sum(image[tot])/(length(image[tot])/2)
    if ( avg_mu <= 0 ) {
      avg_mu = 0
      R_pet[r] = 1
    } else {
      R_pet[r] = mu[r]/avg_mu
    }
    if ( R_pet[r] < eta ) break;
  }
  r_cir = approx(R_pet,1:n,xout=eta)$y
  if ( is.na(r_cir) ) r_cir=n
  return(list(r_cir=r_cir,img_cir=cir))
}

flux_cir2 = function(image,img_cir,r_max)
{
  rhi = as.integer(ceiling(r_max))
  sum_int = rep(0,rhi)
  aver    = rep(0,rhi)

  for ( r in 1:rhi ) {
    if ( r == 1 ) {
      ann = which(img_cir<r+0.5,arr.ind=T)
      sum_int[r] = max(c(sum(image[ann]),0))
    } else {
      ann = which(img_cir>=r-0.5&img_cir<r+0.5,arr.ind=T)
      sum_int[r] = sum_int[r-1] + max(c(sum(image[ann]),0))
    }
    aver[r] = mean(img_cir[ann])
  }
  # 3/25/14: add zero-point to avoid NA output for very concentrated galaxies
  if ( sum_int[rhi] == 0 ) return(list(r20=-9,r80=-9))
  sum_int = c(0,sum_int/sum_int[rhi])
  aver    = c(0,aver)
  r20 = approx(sum_int,y=aver,xout=0.20)$y
  r80 = approx(sum_int,y=aver,xout=0.80)$y
  return(list(r20=r20,r80=r80))
}

conv_image = function(img,sigma)
{
  tx = dim(img)[1]
  ty = dim(img)[2]

  npix = as.integer(5*sigma)
  nx = tx+npix
  ny = ty+npix
  if ( nx %% 2 == 1 ) nx = nx+1
  if ( ny %% 2 == 1 ) ny = ny+1

  psf = array(0.0,c(nx,ny))
  for ( ii in 1:(nx/2) ) {
    for ( jj in 1:(ny/2) ) {
      psf[ii,jj] = exp(-((ii-1)^2+(jj-1)^2)/2/sigma^2)/2/pi/sigma^2
    }
  }
  for ( ii in (nx/2+1):nx ) {
    for ( jj in 1:(ny/2) ) {
      psf[ii,jj] = exp(-((ii-(nx+1))^2+(jj-1)^2)/2/sigma^2)/2/pi/sigma^2
    }
  }
  for ( ii in 1:(nx/2) ) {
    for ( jj in (ny/2+1):ny ) {
      psf[ii,jj] = exp(-((ii-1)^2+(jj-(ny+1))^2)/2/sigma^2)/2/pi/sigma^2
    }
  }
  for ( ii in (nx/2+1):nx ) {
    for ( jj in (ny/2+1):ny ) {
      psf[ii,jj] = exp(-((ii-(nx+1))^2+(jj-(ny+1))^2)/2/sigma^2)/2/pi/sigma^2
    }
  }

  nimg = array(0.0,c(nx,ny))
  for ( ii in 1:tx ) { 
    for ( jj in 1:ty ) { 
      nimg[ii,jj] = img[ii,jj]
    }
  }

  fx = fft(nimg)
  fy = fft(psf)
  cimg = Re(fft(fx*fy,inverse=T))[1:tx,1:ty]/length(fx)

  return(cimg)
}

C_statistic = function(oimg,cimg,r_pet)
{
  r_max = 1.5*r_pet
  if ( r_max > as.integer(floor(min(dim(oimg))*sqrt(2)/2)) ) {
    r_max = as.integer(floor(min(dim(oimg))*sqrt(2)/2))
  }
  r = flux_cir2(oimg,cimg,r_max)
  if ( r$r20 == -9 ) return(-9)
  C = 5*log10(r$r80/r$r20)
  return(C)
}

A_statistic = function(oimg,cimg,smap.all,r_pet)
{
  r_max = 1.5*r_pet
  if ( r_max > as.integer(floor(min(dim(oimg))*sqrt(2)/2)) ) {
    r_max = as.integer(floor(min(dim(oimg))*sqrt(2)/2))
  }
  # Rotate 180 degrees
  rimg = rev(oimg)
  dim(rimg) = dim(oimg)
  w = which(cimg>r_max,arr.ind=T)
  cimg[w] = 0
  w = which(cimg>0,arr.ind=T)
  cimg[w] = 1
  ngal = length(w)/2
  ogal = cimg*oimg
  rgal = cimg*rimg

  Aden = sum(abs(ogal))
  Agal = sum(abs(ogal-rgal)) 

  cimg = cimg-1
  cimg = abs(cimg)
  #oann = cimg*oimg
  oann = cimg*oimg*abs(smap.all-1)
  w = which(oann!=0,arr.ind=T)
  if ( length(w) > 0 ) {
    rann = rev(oann)
    dim(rann) = dim(oann)
    w = which(oann!=0&rann!=0,arr.ind=T)
    if ( length(w) > 0 ) {
      nbkg = length(w)/2
      Abkg = sum(abs(oann[w]-rann[w]))
    } else {
      nbkg = 1
      Abkg = 0
    }
  # nbkg = length(w)/2
  # sann = sort(oann[w])
  # n = length(w)/2
  # m = median(oann[w])
  # while ( m > 0 ) {
  #   wm = which(oann==sann[n],arr.ind=T)
  #   oann[wm] = 0
  #   w = which(oann!=0,arr.ind=T)
  #   if ( length(w) == 0 ) break;
  #   m = median(oann[w])
  #   n = n-1
  # }
  # rann = rev(oann)
  # dim(rann) = dim(oann)
  # Abkg = sum(abs(oann-rann))
  } else {
    nbkg = 1
    Abkg = 0
  }
  A = (Agal-(ngal/nbkg)*Abkg)/(2*Aden) # factor of 2 from Conselice et al. 2000
  return(A)
}

