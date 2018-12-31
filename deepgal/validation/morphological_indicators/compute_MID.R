library(SDMTools)

define_clump = function(img)
{
  x_len = dim(img)[1]
  y_len = dim(img)[2]
  clump = array(-9,c(x_len,y_len))
  x_peak = c()
  y_peak = c()
  for ( jj in 1:x_len ) {
    for ( kk in 1:y_len ) {
      if ( img[jj,kk] != 0.0 ) {
        jjcl = jj
        kkcl = kk
        istop = 0
        while ( istop == 0 ) {
          jjmax = jjcl
          kkmax = kkcl
          imgmax = img[jjcl,kkcl]
          for ( mm in -1:1 ) {
            if ( jjcl+mm > 0 && jjcl+mm <= x_len ) {
              for ( nn in -1:1 ) {
                if ( kkcl+nn > 0 && kkcl+nn <= y_len ) {
                  if ( img[jjcl+mm,kkcl+nn] > imgmax ) {
                    imgmax = img[jjcl+mm,kkcl+nn]
                    jjmax = jjcl+mm
                    kkmax = kkcl+nn
                  }
                }
              }
            }
          }
          if ( jjmax == jjcl && kkmax == kkcl ) {
            n = length(x_peak)
            if ( n > 0 ) {
              cltmp = 0
              for ( pp in 1:n ) {
                if ( x_peak[pp] == jjmax && y_peak[pp] == kkmax ) {
                  ifound = 1
                  cltmp = pp
                }
              }
              if ( cltmp > 0 ) {
                clump[jj,kk] = cltmp
              } else {
                x_peak = append(x_peak,jjmax)
                y_peak = append(y_peak,kkmax)
                clump[jj,kk] = length(x_peak)
              }
            } else {
              x_peak = append(x_peak,jjmax)
              y_peak = append(y_peak,kkmax)
              clump[jj,kk] = length(x_peak)
            }
            istop = 1
          } else {
            jjcl = jjmax
            kkcl = kkmax
          }
        }
      }
    }
  }
  return(list(clump=clump,x_peak=x_peak,y_peak=y_peak))
}

#smooth_data = function(img,scale)
#{
#  smooth_img = array(0.0,dim(img))
#  boxsize = ceiling(3*max(scale))
#  if ( boxsize%%2 == 0 ) boxsize = boxsize+1
#  for ( jj in ceiling(boxsize/2):(dim(img)[1]-floor(boxsize/2)) ) {
#    for ( kk in ceiling(boxsize/2):(dim(img)[2]-floor(boxsize/2)) ) {
#      if ( img[jj,kk] != 0 ) {
#        totwght = 0
#        for ( mm in -floor(boxsize/2):floor(boxsize/2) ) {
#          for ( nn in -floor(boxsize/2):floor(boxsize/2) ) {
#            if ( img[jj+mm,kk+nn] != 0 ) {
#              dist2 = mm^2+nn^2
#              wght = exp(-dist2/2/scale^2)/(2*pi)/scale^2
#              smooth_img[jj,kk] = smooth_img[jj,kk]+img[jj+mm,kk+nn]*wght
#              totwght = totwght+wght
#            }
#          }
#        }
#        smooth_img[jj,kk] = smooth_img[jj,kk]/totwght
#      }
#    }
#  }
#  return(smooth_img)
#}

M_statistic = function(img,levels=seq(0.0,0.975,by=0.025))
{
  npix = dim(img)[1]*dim(img)[2]-length(which(img==0))
  norm_img = img/max(img)
  nlevels = length(levels)
  area_ratio = rep(0.0,nlevels)
  area_ratio_o = rep(0.0,nlevels)
  area_ratio_p = rep(0.0,nlevels)
  max_level = 0
  max_level_o = 0
  max_level_p = 0

  w = which(norm_img!=0,arr.ind=T)
  v = sort(as.vector(norm_img[w]))

  for ( ii in 1:nlevels ) {
    thr = round(npix*levels[ii])
    if ( thr <= 1 ) { next; }
    # determine 8-connected clumps in remaining non-zero pixels
    clump = ConnCompLabel(norm_img>=v[thr])
    if ( length(clump) == 0 ) { next; }
    if ( is.na(clump[1,1]) ) { next; }
    if ( sum(clump) == 0 ) { next; }
    areas = suppressMessages(ClassStat(clump))$total.area[-1]
    if ( length(areas) > 1 ) {
      areas = sort(areas,decreasing=T)
      area_ratio[ii] = (areas[2]/areas[1])*(areas[2]/npix)
      area_ratio_o[ii] = (areas[2]/areas[1])*areas[2]
      area_ratio_p[ii] = areas[2]/areas[1]
    }
  }
  if ( max(area_ratio) > 0 ) max_level = levels[which.max(area_ratio)]
  if ( max(area_ratio_o) > 0 ) max_level_o = levels[which.max(area_ratio_o)]
  if ( max(area_ratio_p) > 0 ) max_level_p = levels[which.max(area_ratio_p)]
  return(list(level=max_level,level_o=max_level_o,level_p=max_level_p,M=max(area_ratio),M_o=max(area_ratio_o),M_p=max(area_ratio_p)))
}

I_statistic = function(img,scale=0)
{
# smooth_img = smooth_data(img,scale)
  if ( scale > 0 ) {
    smooth_img = conv_image(img,scale)
    w = which(abs(smooth_img)<1.e-5,arr.ind=T)
    smooth_img[w] = 0
  } else {
    smooth_img = img
  }
  clump = define_clump(smooth_img)
  n = length(clump$x_peak)
  if ( n == 1 ) {
    intensity_ratio = 0.0
    x_peak = (clump$x_peak)[1]
    y_peak = (clump$y_peak)[1]
  } else {
    clump_intensity = rep(0.0,n)
    for ( jj in 1:(dim(img)[1]) ) {
      for ( kk in 1:(dim(img)[2]) ) {
        if ( clump$clump[jj,kk] > 0 ) {
          clump_intensity[clump$clump[jj,kk]] =
            clump_intensity[clump$clump[jj,kk]]+img[jj,kk]
        }
      }
    }
    w = which.max(clump_intensity)
    x_peak = (clump$x_peak)[w]
    y_peak = (clump$y_peak)[w]
    s = sort(clump_intensity,decreasing=T)
    intensity_ratio = s[2]/s[1]
  }
  return(list(intensity_ratio=intensity_ratio,x_peak=x_peak,y_peak=y_peak))
}

D_statistic = function(img,x_peak,y_peak)
{
  if ( is.null(x_peak) || is.null(y_peak) ) return(-9)
  # Center of mass
  tot   = 0
  x_cen = 0
  y_cen = 0
  for ( jj in 1:(dim(img)[1]) ) {
    for ( kk in 1:(dim(img)[2]) ) {
      if ( img[jj,kk] > 0 ) {      # ignore negative pixels for CoM computation
        x_cen = x_cen+jj*img[jj,kk]
        y_cen = y_cen+kk*img[jj,kk]
        tot   = tot+img[jj,kk]
      }
    }
  }
  x_cen = x_cen/tot
  y_cen = y_cen/tot
  area = length(which(img!=0))
  deviation = sqrt((x_peak-x_cen)^2+(y_peak-y_cen)^2)/sqrt(area/pi)
  return(deviation)
}
