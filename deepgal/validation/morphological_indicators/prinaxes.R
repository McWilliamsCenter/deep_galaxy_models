#
# assumes image is *normalized to 1*
#
prinaxes = function(img,debug=F)
{
  x = 1:(dim(img)[1])
  y = 1:(dim(img)[2])
  marx = apply(img,1,sum)
  mary = apply(img,2,sum)
  w = which(marx<0)
  marx[w] = 0.0
  w = which(mary<0)
  mary[w] = 0.0
  xcen = sum(x*marx)
  ycen = sum(y*mary)
  xvar = sum(x*x*marx)-xcen^2
  yvar = sum(y*y*mary)-ycen^2
  xycov = sum((x%*%t(y))*img)-xcen*ycen
#
  b = -xvar-yvar
  c = xvar*yvar-xycov^2
  sd = sqrt(b**2-4*1*c)
  axmin = sqrt(0.5*(-b-sd))
  axmax = sqrt(0.5*(-b+sd))
  rad2deg = 180/pi
  a = xvar-axmax^2
  b = xycov
  c = yvar-axmax^2
  angle = atan((a-b)/(c-b))*rad2deg
  return(list(axmax=axmax,axmin=axmin,angle=angle))
}

