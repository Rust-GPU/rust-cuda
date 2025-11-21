# Denoiser

Example of a very simple binary which loads an image from a path and runs it through the OptiX 
AI denoiser to denoise it and output it in the same directory.

Note that this is just about the worst way to run OptiX because the input is not Hdr, and it provides
no albedo and normal guide images, so results may be pretty bad around object edges and such.
