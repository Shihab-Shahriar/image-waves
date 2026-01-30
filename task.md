Task: Process video horizontal waves to find wavelength

Phase 1: The Setup (Data Preparation)
1. Let Users define the Region of Interest (ROI) from a video (maybe by selecting oen frame as camera doesn't move)

2. Hardcode a $M$ of diagram notes $M = \frac{mm}{Pix}$, a real-world conversion factor for now


Phase 2: The Core Algorithm (Per Frame)
3. "Collapse" the Image (The "Y-Average")This is the most critical step shown in the bottom half of your drawing.Concept: You have a 2D box of pixels. The waves are vertical lines. If you look at just one horizontal row of pixels, it might be grainy. But if you average all the rows together, the graininess cancels out, and the wave pattern stands out.Action: For every column ($x$) in your ROI, calculate the average brightness of all pixels in that vertical strip ($y$).Output: You now have a simple array of numbers (a 1D waveform) representing light and dark intensity across the image.
4. Smoothing (Noise Reduction)Even after averaging, the line might be "jagged."Action: Apply a simple "Moving Average" or Gaussian blur to your 1D array.Goal: Turn a jagged, shaky line into a smooth, rolling wave. This prevents the computer from mistaking a tiny speck of dust for a wave crest.
   
5.  Peak/Valley DetectionYour diagram shows dips (valleys) marked as $L_1, L_2$.Action: Scan through your 1D array to find the "local minima" (the lowest points). These represent the center of the dark wave lines.Logic: A point is a local minimum if it is lower than the neighbors to its left and right.
   
6.  Calculate Pixel WavelengthAction: Calculate the distance (in pixels) between consecutive minima ($x_2 - x_1$, $x_3 - x_2$, etc.).Averaging: Real waves aren't perfect. Don't rely on just one pair. Calculate the distance between all detected pairs in the frame and average them to get a single Average_Pixel_Wavelength.7. Convert to Physical UnitsFinal Math: Real_Wavelength = Average_Pixel_Wavelength * Scale_Factor.

7. Convert to Physical Units

Final Math: Real_Wavelength = Average_Pixel_Wavelength * Scale_Factor. 


Key Design Decisions:
How deep must a "dip" be to count as a wave? If you set the threshold too low, shadows and scratches on the glass will look like waves. If you set it too high, you might miss faint waves.Recommendation: Use a "prominence" setting—a peak must be $X$% darker than the surrounding average to count. 

Handling "Half Waves"

What happens if a wave is just entering the frame on the left? It might look like a valley but have no left neighbor.

Decision: Explicitly ignore edge cases. Only count a wave if you can see the full "dip" (down and back up) to ensure accuracy.

Temporal Averaging (Time vs. Space)

diagram shows spatial averaging (collapsing Y). Since this is a video, you can also average across time (frames). For now stick to one frame. 