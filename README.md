# FMCW radar processing

This tkinter app can be used to process fmcw radar data and its main purpose is to be able to change processing parameters and methods on the fly.

The data is organised this way: 
(frames, antennas signals, samples)
and is processed considering two following antennas signals are the i/q components of a single antenna signal. The script only takes up to two antennas for now.

The app produces a range-doppler map and various variants of it by passing it in several filters ending with a list of maxima that are used to estimate the range, speed an angle of some target(s). From these we can extract the 2D position and apply filtering/tracking algorithms.

The processing is organized this way:<br />
<p style="text-align: center;">raw i+j*q signal <br />
&#8595;<br />delete pauses between chirps<br />
&#8595;<br />chirp matrix form<br />
&#8595;<br />delete zero velocities (offset in columns)<br />
&#8595;<br />windowing<br />
&#8595;<br />ifft2<br />
&#8595;<br />Correlation function<br />
&#8595;<br />noise filtering<br />
&#8595;<br />peak correlation filtering<br />
&#8595;<br />maxima finding<br />
&#8595;<br />range, speed and angle estimates<br />
&#8595;<br />target association<br />
&#8595;<br />target data filtering/tracking
</p>
