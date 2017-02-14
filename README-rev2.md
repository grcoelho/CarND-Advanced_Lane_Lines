Project: Advanced Lane Finding - Second Review
===================


After the first review, I worked on how to get the output more stable during the bridge crossing (where the tormac becames brighter).

To do that I implemented the following:

#### <i class="icon-file"></i> Frames Drop
This function received the calculated parameters of the left and right curves, and compare it with the last valid frame.
The decision was made in terms of the square parameter A. I tried with average of all and with only B and C, but got better results only using the A parameter. 
If A deviates more than *abc_error_limit* that was tested with good results with 1.15e-04, the frame will be ignored and the last calculations will be reused.
Since A defines how sharp the curve is, if the curve changes in angle from one frame to the other by more than the factor define, it will be ignored.

To avoid a big number of frames to be ignored, a limiter was set. Since the video is 25 fps, I defined that no more than 25 frames can be drop without new calculations. So the final video wont be more than 1 second without curve updates.

```
def ignore_data_or_not(left_fit,right_fit):
    global number_of_stored_frames
    global abc_error_limit
    global historic
    global frame
    global debug
    global drop_frames
    global drop_frames_resetable
    global maximum_ignored_frames
    ignore = 0
    #Checking that they have similar curvature
    #Checking that they are separated by approximately the right distance horizontally
    #Checking that they are roughly parallel
    
    #determine the data deviation from the last 6 frames
    if (frame==0) or (drop_frames_resetable>maximum_ignored_frames):
        #reset the matrix and just store data
        historic = np.zeros((3, 2*number_of_stored_frames))
        #if the program is here and its not the first frame, we should reset it again
        drop_frames_resetable = 0 
        historic [:,0] = left_fit
        historic [:,1] = right_fit
        ignore = 0
        if debug==1:
            print("just store data")
            print(historic)
        return left_fit, right_fit, ignore
    else:
        #compare with the last frame
        abc_error_left = abs(historic [0,0] - left_fit[0])# + abs(historic [1,0] - left_fit[1]) + abs(historic [2,0] - left_fit[2])
        abc_error_right = abs(historic [0,1] - right_fit[0])# + abs(historic [1,1] - right_fit[1]) + abs(historic [2,1] - right_fit[2])
        if (abc_error_left<abc_error_limit) and (abc_error_right<abc_error_limit):
            #ok, the difference is tolerable, shift and store this frame
            historic = np.roll(historic, 2, axis=1)
            historic [:,0] = left_fit
            historic [:,1] = right_fit
            ignore = 0          
            if debug==1:
                print("frame not 1, error ok. not ignored. stored")
                print(historic)
            return left_fit, right_fit, ignore
            #if (frame>6):
            #    left_fit_original = historic[:, 1::2]
            #    right_fit_original = historic[:, ::2]

            #    left_fit_mean = np.mean(left_fit_original, axis=1)
            #    right_fit_mean = np.mean(right_fit_original, axis=1)
            #    return left_fit_mean, right_fit_mean, ignore
            #else:
            #    return left_fit, right_fit, ignore
```


#### <i class="icon-file"></i> Histogram Equalization

I read in slack that Histogram Equalization was a good way to smooth the changes in contrast in images. I found [this](http://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html) article interesting.
I used first the function *cv2.equalizeHist()* but found that the result wasnt significative.
But when trying with CLAHE (Contrast Limited Adaptive Histogram Equalization), I found that the results from Sobel had a great improvement, specifically on the dark part of the road (where earlier it was not so great).
But my problems were still laying around the brighter parts of the lane, so I cut this section of the video and start working only on it.

#### <i class="icon-file"></i> Color Parameters Adjustments	

Since now sobel is giving a more acurate picture on dark lanes, I decided to tune the masking algorithm to saturate in higher levels. So now it is giving almost no information on the dark lanes, but its giving a better pictures on brighter lanes. On the video that I put both pictures side by side, its possible to see this effect. Basically sobel drives on dark lanes, and masking drives on brighter lanes.

#### <i class="icon-file"></i> Window Centroids

Looking at my generated images, I found out that the dashed lanes put a problem to polyfit. Sometimes I had no more than 1 dash into the picture, and the polyfit wasnt able to fit a curve into it (just a straight line). In other examples, the detection was not so clear, so polyfit wasnt able to join the segments.
Then I implemented a function to fit windows of adjusted size to these points or dashes. Trying to increase the amount of information available to polyfit.
Since the dashes sometimes are far away from each other, I had to use big windows. This implies a lack of resolution that can result in some errors too, but I couldnt find another way to get more information from the dashes than using these large windows.
The result of this function is a picture with white windows that stacked form two lines. This picture is then feed into the polyfit original function.

#### <i class="icon-file"></i> Lane Angles

Like it was mentioned in the first review, I forgot to convert the info from the lanes angles from pixels to meters. While running the video they appear to be correct to me.
I came back to the lesson and found the mistake, I had to fit a new polygon to it in "meters space", after converting from pixels to meters. I also used new estimations for the lane width to calculate pixels per meter in x direction, and used dash lenght to calculate for y direction.
```
y_eval = np.max(ploty)
    #left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    #right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    #fit a second polynom **** After review
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad
```

#### <i class="icon-file"></i> Results

Running the video, I think that it is behaving better over the bridges. The frame drop algorithm is killing the false results and calculating when the video processing algorithms give a better picture of the lane. Its easy to see this behavior because the algorithm is reacting a bit slower than earlier to changes in the lanes.
I noticed that the algorithm performs a little worse on dashed lanes, and I think this could be due to the Window Centroids that now gives information on the dashes, but still uses a low resolution to do that, and I think that earlier I had Masking algorithm to deal with the darker lanes, and now its tunned to work better on brighter lanes. As the debug video shows, the dashed lines are not seem on the masked video (middle video on the right), but I wasnt able to tune it in a way that the dashes appear on the left and the masking doesnt get crazy on the brighter lanes.
I spent some time working on these parameters, and couldn't find a way to make it work perfect on both lanes. I know it is still far from perfect, but I think that this new solution is clever (to ignore possible wrong frames) and the final result seems steadier. 

Like the reviewer said, I tried to use past information to get a better result. But in my experiments (I tried to work on the average of the last 6 frames, when data is considered wrong, but it seems far better using just info from last frame and forcing it to calculate at each bunch of frames ignored).


Project: Advanced Lane Finding - First Review
===================


The goals of this project are:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

To accomplish that, I read the article from Vivek Yadav on Medium:
[Robust lane finding using advanced computer vision techniques: Mid project update](https://medium.com/@vivek.yadav/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3#.n4h6jf4ka)

I used Vivek's tips and most of the code explained in the Advanced Lane Finding lesson to build a model that works on the project video. The result can be seen [here](https://www.youtube.com/watch?v=7EziU8Gxmaw) on Youtube.


Step-by-Step Approach
-------------


I decided to use a Jupyter Notebook for this project, like I was using on the last ones. All the code is in this notebook, and I made some coments explained what I tought.

#### <i class="icon-file"></i> Camera Calibration

The first thing is to calibrate the camera. Like explained in the class, I loaded all images inside the camera_cal directory using glob. The images inside this folder had 9 by 6 edges on each chessboard.
```
images = glob.glob('camera_cal/calibration*.jpg')
edge_x = 9
edge_y = 6
```
For each image I run the corner detection algorithm 
```
ret, corners = cv2.findChessboardCorners(gray, (edge_x,edge_y),None)
```
And generated the calibration parameters
```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
```
Here is an example of chessboard before and after the calibration.

![enter image description here](https://lh3.googleusercontent.com/G0wTrwgYI1v5MbUc9O_n6hEbr7TV4xdisz8eE2kNF80UBaD8HorDJXFlMgscC-ydpgqOidplIw=s0 "camera_calib.png")

#### <i class="icon-file"></i> Image Processing
The core of the code is inside the *process_image* function. First it applies the undistort function, like we used on the class.
```
img = cv2.undistort(img_original, mtx, dist, None, mtx)
```
Then it transforms the original image into a perspective (Birds-Eye view) using:
```
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```
To define the destination points (dst) I just fixed a square into the image
```
dst = np.float32(
        [[0,img_y],
         [img_x,img_y],
         [0,0],
         [img_x,0]])
```
The source points (src) were a bit more difficult to determine. I decided that I should be able to adjust them easily. So I set a function to take points using parameters like: Percentage of horizon removal, percentage of left removal and right removal (for both, upper and lower pair of points).
```
	#Birds eye view transformation parameters
	# p = [x,y] from the upper left corner
	bonnet = 40 # pixels were removed from y to take the bonnet out
	horizon = 0.35    # percentage of the picture to remove the horizon
	sides_low = 0.13  # percentage of the picture to remove on the lower points
	sides_up = 0.39   # percentage of the picture to remove on the upper points
    p1 = [int(img_x*sides_low),img_y-bonnet] 
    p2 = [img_x - int(img_x*sides_low),img_y-bonnet] 
    p3 = [int(img_x*sides_up),img_y-int(img_y*horizon)] 
    p4 = [img_x - int(img_x*sides_up),img_y-int(img_y*horizon)] 
```

The result is the four points on the following picture:
![enter image description here](https://lh3.googleusercontent.com/t69hNLshgA23M0OXDSwEKU8pW72I_z_whKharZ7nHMNEPallRHLLOF6bYxhaekkongbXhOnIyg=s0 "points.png")

After the transformation, the image appears like the following one:

![enter image description here](https://lh3.googleusercontent.com/wLXTZ_dPgwfeqtHDS7rM55c8CpSZUghGbvuGLH0OIvB0S0UfgOeAP7NsFoHT9sS3DPpyunTUtA=s0 "birdseye.png")

With this image, I applied Gaussian Blurring (to remove noise) and then Masked the Yellow and White lines. 
```
    #Apply Gaussian blur to the bird eye view image
    warped = cv2.GaussianBlur(warped,(gaussian_kernel, gaussian_kernel), 0)
    hsv = cv2.cvtColor(warped,cv2.COLOR_RGB2HSV)
    # Threshold the HSV image to get only yellow lines
    mask_y = cv2.inRange(hsv, low_yellow, high_yellow)
    # Bitwise-AND mask and original image
    res_y = cv2.bitwise_and(warped,warped, mask= mask_y)
    # apply the mask on the original image
    hsv_mask_yellow = cv2.inRange(hsv, low_yellow, high_yellow)
    # Threshold the HSV image to get only blue colors
    mask_w = cv2.inRange(hsv, low_white, high_white)
    # Bitwise-AND mask and original image
    res_w = cv2.bitwise_and(warped,warped, mask= mask_w)
    # apply the mask on the original image
    hsv_mask_white = cv2.inRange(hsv, low_white, high_white)
```
The thresholds were defined in HSV color space using try and error method in a series of training images:
```
#Manually Tunning the masks to the images
low_yellow  = np.array([ 0,  100,  100])
high_yellow = np.array([ 50, 255, 255])
low_white  = np.array([ 20,   0,   180])
high_white = np.array([ 255,  80, 255])
```
The results are as follows:

![enter image description here](https://lh3.googleusercontent.com/oaA1XYhsGRcAHtyst1x80dWToiXkyMsNuu-59AqC0VnxAkmtVRUB8TiFMfRiflRSaTlNIjjG-Q=s0 "mask.png")

I also used Sobel transform to get more information from the original image. I used the 3 functions explained in class for ABS, MAG and DIR to see wich one gives me the best result. I ended up deciding to keep only the ABS because of its lower noise level.

![enter image description here](https://lh3.googleusercontent.com/FY9nhEohgF3LBtBzlItWkvFzt8Mfxv9Gt8PYhWlnVzM4hXXoq7FTaGeB72JnUC7jNDutAFMfXw=s0 "sobel.png")

I merged sobel ABS with the Maks obtained from previous processing, and used them to further fit the polygon curves.

![enter image description here](https://lh3.googleusercontent.com/MacIYe7n2Fnb33CQ2Pxhi2DGbExCMpsFoqFUuXczuWDsgp_OEQN3e-gmv4AW0MHXLN2M6aKAuA=s0 "sobel_mask.png")

This image was sent to *polynom_fitting*, that is the same polyfit function explained in the class. 
I divided it in two. The first one process all the picture looking for lines, while the second one uses pre-processed info to look only in a 100px radius of the last curve. This is done to avoid processing all the image in each frame (it also helps reducing noise, since the likehood area is narrowed).
```
def polynom_fitting(binary_warped):
    global frame
    #Same code used in the class.
    #input: birds eye view image
    #output: image with function plotted and polynom parameters
    print("frame: ",frame)
    if frame == 0:
        print("inside first frame")
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        #global left_fit
        #global right_fit
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        print ("left fit: ", left_fit)
        print("Right Fit: ",right_fit)
        

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Create empty lists to receive left and right lane pixel indices
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        global last_left_fit
        global last_right_fit
        
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (last_left_fit[0]*(nonzeroy**2) + last_left_fit[1]*nonzeroy + last_left_fit[2] - margin)) & (nonzerox < (last_left_fit[0]*(nonzeroy**2) + last_left_fit[1]*nonzeroy + last_left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (last_right_fit[0]*(nonzeroy**2) + last_right_fit[1]*nonzeroy + last_right_fit[2] - margin)) & (nonzerox < (last_right_fit[0]*(nonzeroy**2) + last_right_fit[1]*nonzeroy + last_right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = last_left_fit[0]*ploty**2 + last_left_fit[1]*ploty + last_left_fit[2]
        right_fitx = last_right_fit[0]*ploty**2 + last_right_fit[1]*ploty + last_right_fit[2]


    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    #calculate the curvature radius
    return result, left_fitx, right_fitx, left_fit, right_fit, ploty, leftx, rightx
```
The result is a polygon fitted between the two curves (plotted using the info from the masked+sobel image).
Here are two images, one from the curve yet in the birds-eye view and the other one is from the polygon already transformed to the original perspective.

![enter image description here](https://lh3.googleusercontent.com/L2lcKFj7RxhWlO3IeJfFy1cmAF2p2u2aUMfyQstBO1u7OVIjQkh9ds1aAny3sqfjpWv5N9f_iQ=s0 "curves.png")

![enter image description here](https://lh3.googleusercontent.com/l6jcxQ8vsn2ixnOb8g-OSOF4g7JBTNZMWH4qlVLowjq_e1f1tBljWdaZNH4RIEtzsNx2HkHgvA=s0 "polygon.png")

Then I just merged that with the original image.
I also used the function explained in the class to get the curvature radius.
```
def calculate_centerline_turn (leftx,rightx,left_fit,right_fit,ploty):
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad
```
Later I developed a function to calculate the lane deviation from center.
This just measure the center of the polygon and the center of the original picture. If the result is positive, it means that the car is Right from the lane, and if negative means it deviated to the Left of the lane.
```
def find_center(image):
    for y in range(719, 0, -1):
        for x in range(1279, 0, -1):    
            if image[y,x,0] == 255:
                #ok, I found red in line y in this image
                for x1 in range(1279, 0, -1): 
                    if image[y,x1,0] == 255:
                        bigger_x = x1
                        break
                for x1 in range(0,1279): 
                    if image[y,x1,0] == 255:
                        smaller_x = x1
                        track_center_x = 1280/2
                        poly_center = smaller_x + ((bigger_x - smaller_x)/2)
                        #Pixels out center - If positive, vehicle on the right of the lane / Negative: vehicle to the left of lane 
                        out_center = track_center_x-poly_center
                        #convert these pixels to meters
                        out_center_cm = 100*xm_per_pix * out_center
                        return out_center_cm
```                    
These data are ploted to the top left corner of the original image.

![enter image description here](https://lh3.googleusercontent.com/sW_2-T7lZh-6KcuJhfEwJqYy1HzBIHgjB4pO3X06pwg6LHxTswuupjnwSCvkJAyBeGk3sTXVFA=s0 "merged_final.png")


#### <i class="icon-file"></i> Video Processing
All the code was condensed into the *process_frame* function. So to process the video, I just open the capture of each frame, apply the *process_frame* function and get the image already including the red lane and the data from deviation and radius.
All these are save in a new video file.
```   
while(cap.isOpened()):
    ret, img_original = cap.read()
    if ret==True:
        lane, last_left_fit, last_right_fit = process_frame (img_original)
        out.write(lane)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("Finished")
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
```   

Reflections
-------------
From what I could observe, the code behaves well on the *project_video.mp4*, but it fails on the other videos. I blame that on the fine tunning needed for this code to work.
All the parameters like region for image perspective transformation and the mask/sobel parameters were tunned to work on these brightness and shadowing conditions. The lane line colors were also tunned to fit the example.
For this code to be really adaptive, it should evaluate the lane line brightness and the color of the lines in a dynamic way. It will also fail if the car changes lanes for example (that is something really common). 

So the proposed problem is still pretty much something that works only in specific conditions and will not fit everyday problems (For example, in my country Bus Lanes are marked in blue color, not yellow or white. Our car will fail to detect that with some simple algorithm like the proposed on the lesson and will end up driving into the bus lane).

I feel that this is good for training purposes, but the combination of image processing and neuro networks may work better. I want to try it in the future, but unfortunatelly I dont have time to research on that before the project deadline.
