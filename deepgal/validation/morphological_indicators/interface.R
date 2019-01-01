source("./prinaxes.R")
source("./segmap.R")
source("./compute_CA.R")
source("./compute_GM20.R")
source("./compute_MID.R")

compute_statistics_single <- function(img,      # 2D image
                               scale.pix=0.03,    # Pixel scale in arcsec
                               scale.smooth=1,    # Smoothing scale for I and D
                                                  # statistics, in pix
                               eta=0.2,           # Segmentation parameters
                               thrlev=10          # Segmentation parameters
                               ) {
     # Initializing variables
     M       <- M_o        <- M_p       <- -9
     M_level <- M_level_o  <- M_level_p <- -9
     I       <- D          <- -9
     Gini    <- M20        <- -9
     C       <- A          <- -9
     axmax   <- axmin      <- angle <- -9
     sn      <- -9
     size    <- -9

    result <- tryCatch({
        # Check for zeroes in the postage stamp image, indicating
        # possible edge-of-field issues.
        w <- which(img == 0, arr.ind = TRUE)
        if ( length(w[, 1]) / length(img) > 0.05 ) next;  # ad hoc

        # Create segmentation map (mask) based on algorithm of Freeman et al
        out <- segmap(img, eta = eta, thrlev = thrlev)
        smap <- out$c
        smap.all <- out$allc  # for A computation (and adding noise)

        w <- which(smap > 0, arr.ind = T)
        r_smap <- sqrt(length(w) / 2 / pi)
        if ( is.null(smap) == T ) next;

        # Estimator of galaxy "size" (effective "radius")
        w <- which(smap > 0, arr.ind = T)
        size <- sqrt(length(w) / 2 / pi)

        # My estimator of signal-to-noise (S/N)
        w <- which(smap.all == 0, arr.ind = T)
        muhat <- mean(img[w], trim = 0.05)
        sdhat <- sd(img[w])
        z <- c()
        for ( jj in 1:dim(img)[1] ) {
          for ( kk in 1:dim(img)[2] ) {
            if ( smap[jj, kk] > 0 ) {
              if ( sqrt( (jj - (dim(img)[1]) / 2) ^ 2 +
                         (kk - (dim(img)[2]) / 2) ^ 2 ) < 0.5 * size ) {
                z <- append(z, ( img[jj, kk] - muhat) / sdhat)
              }
            }
          }
        }
        if ( length(z) == 0 ) next;
        sn <- median(z)

        # Convert size from pixels to arc-seconds
        size <- scale.pix * size

        # Zero out all pixels outside the mask
        oimg   <- img
        img    <- img * smap
        w      <- which(img < 0, arr.ind=T)
        img[w] <- 0

        # Normalize the image
        nimg <- img / sum(img)

        # Compute principal axes: semi-major/semi-minor axes, rotation angle
        pa <- prinaxes(nimg)
        axmax <- pa$axmax
        axmin <- pa$axmin
        angle <- pa$angle

        # Compute M statistic of Freeman et al. (M_o) plus two variants
        out = M_statistic(img, levels = seq(0.00, 0.99, by = 0.025))

        M <- out$M
        M_o <- out$M_o
        M_p <- out$M_p
        M_level <- out$level
        M_level_o <- out$level_o
        M_level_p <- out$level_p

        # Compute I and D statistics of Freeman et al.,
        # with pre-smoothing of data to mitigate noise (kernel size ~ 1 pix)
        out <- I_statistic(img, scale = scale.smooth)
        I <- out$intensity_ratio
        D <- D_statistic(img, out$x_peak, out$y_peak)

        # Compute Gini and M_20 based on prescription of Lotz et al. (2004)
        Gini <- Gini_statistic(img)
        M20  <- M20_statistic(img)

        # Compute CA (Conselice 2003)
        xc <- dim(oimg)[1] / 2
        yc <- dim(oimg)[2] / 2
        if ( xc %% 2 == 0 ) xc <- xc + 0.5
        if ( yc %% 2 == 0 ) yc <- yc + 0.5
        cir   <- pet_rad_cir2(oimg, xc, yc)
        cimg  <- cir$img_cir
        r_pet <- cir$r_cir
        if ( r_pet > 2 * r_smap ) r_pet <- r_smap
        C <- C_statistic(oimg, cimg, r_pet)
        A <- A_statistic(oimg, cimg, smap.all, r_pet)
      },
        error = function(c) {
          return(FALSE)
        }
      )

      res <- data.frame(M_level, M, M_level_o, M_o, M_level_p, M_p,
                        I, D, axmax, axmin, angle, sn, size, Gini, M20,
                        C, A)
      if ( result == FALSE ) {
          return(list(finish = FALSE, stats = res))
      }
      return( list(finish = TRUE, stats = res) )
}

compute_statistics <- function(images,            # 3D array of images
                               scale.pix=0.03,    # Pixel scale in arcsec
                               scale.smooth=1,    # Smoothing scale for I and D
                                                  # statistics, in pix
                               eta=0.2,           # Segmentation parameters
                               thrlev=10,         # Segmentation parameters
                               seed.user=101){
    nimg <- dim(images)[1]

    # Initializing variables
    M       <- M_o        <- M_p       <- rep(-9, nimg)
    M_level <- M_level_o  <- M_level_p <- rep(-9, nimg)
    I       <- D          <- rep(-9, nimg)
    Gini    <- M20        <- rep(-9, nimg)
    C       <- A          <- rep(-9, nimg)
    axmax   <- axmin      <- angle <- rep(-9, nimg)
    sn      <- rep(-9, nimg)
    size    <- rep(-9, nimg)
    id      <- 1:nimg

    # Sets random seed I guess
    set.seed(seed.user)

    for ( ii in 1:nimg ) {

        result <- tryCatch({
            # Retrieve image from input image list
            img <- images[ii, , ]

            # Check for zeroes in the postage stamp image, indicating
            # possible edge-of-field issues.
            w <- which(img == 0, arr.ind = TRUE)
            if ( length(w[, 1]) / length(img) > 0.05 ) next;  # ad hoc

            # Create segmentation map (mask) based on algorithm of Freeman et al
            out <- segmap(img, eta = eta, thrlev = thrlev)
            smap <- out$c
            smap.all <- out$allc  # for A computation (and adding noise)

            w <- which(smap > 0, arr.ind = T)
            r_smap <- sqrt(length(w) / 2 / pi)
            if ( is.null(smap) == T ) next;

            # Estimator of galaxy "size" (effective "radius")
            w <- which(smap > 0, arr.ind = T)
            size[ii] <- sqrt(length(w) / 2 / pi)

            # My estimator of signal-to-noise (S/N)
            w <- which(smap.all == 0, arr.ind = T)
            muhat <- mean(img[w], trim = 0.05)
            sdhat <- sd(img[w])
            z <- c()
            for ( jj in 1:dim(img)[1] ) {
              for ( kk in 1:dim(img)[2] ) {
                if ( smap[jj, kk] > 0 ) {
                  if ( sqrt( (jj - (dim(img)[1]) / 2) ^ 2 +
                             (kk - (dim(img)[2]) / 2) ^ 2 ) < 0.5 * size[ii] ) {
                    z <- append(z, ( img[jj, kk] - muhat) / sdhat)
                  }
                }
              }
            }
            if ( length(z) == 0 ) next;
            sn[ii] <- median(z)

            # Convert size from pixels to arc-seconds
            size[ii] <- scale.pix * size[ii]

            # Zero out all pixels outside the mask
            oimg   <- img
            img    <- img * smap
            w      <- which(img < 0, arr.ind=T)
            img[w] <- 0

            # Normalize the image
            nimg <- img / sum(img)

            # Compute principal axes: semi-major/semi-minor axes, rotation angle
            pa <- prinaxes(nimg)
            axmax[ii] <- pa$axmax
            axmin[ii] <- pa$axmin
            angle[ii] <- pa$angle

            # Compute M statistic of Freeman et al. (M_o) plus two variants
            out = M_statistic(img, levels = seq(0.00, 0.99, by = 0.025))

            M[ii] <- out$M
            M_o[ii] <- out$M_o
            M_p[ii] <- out$M_p
            M_level[ii] <- out$level
            M_level_o[ii] <- out$level_o
            M_level_p[ii] <- out$level_p

            # Compute I and D statistics of Freeman et al.,
            # with pre-smoothing of data to mitigate noise (kernel size ~ 1 pix)
            out <- I_statistic(img, scale = scale.smooth)
            I[ii] <- out$intensity_ratio
            D[ii] <- D_statistic(img, out$x_peak, out$y_peak)

            # Compute Gini and M_20 based on prescription of Lotz et al. (2004)
            Gini[ii] <- Gini_statistic(img)
            M20[ii]  <- M20_statistic(img)

            # Compute CA (Conselice 2003)
            xc <- dim(oimg)[1] / 2
            yc <- dim(oimg)[2] / 2
            if ( xc %% 2 == 0 ) xc <- xc + 0.5
            if ( yc %% 2 == 0 ) yc <- yc + 0.5
            cir   <- pet_rad_cir2(oimg, xc, yc)
            cimg  <- cir$img_cir
            r_pet <- cir$r_cir
            if ( r_pet > 2 * r_smap ) r_pet <- r_smap
            C[ii] <- C_statistic(oimg, cimg, r_pet)
            A[ii] <- A_statistic(oimg, cimg, smap.all, r_pet)
          },
            error = function(c) {
              return(FALSE)
            }
          )

      if ( result == FALSE ) {
          return(list(finish = FALSE, image.number = ii))
      }
    }

    res <- data.frame(M_level, M, M_level_o, M_o, M_level_p, M_p,
                      I, D, axmax, axmin, angle, sn, size, Gini, M20,
                      C, A)
    return( list(finish = TRUE, stats = res) )
}
