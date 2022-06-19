% show how was data is loaded from single collection period, along with subplot 
% used to help me determine how much of the data to use from start to stop 
% since there were anomalies in data when user adjusts phone to start/stop
% collection

load('C:\Users\Nader\MATLAB Drive\MobileSensorData\drive-withwalk-49mins.mat')
subplot(311),plot(Acceleration.X)% this subplot allowed me to see where to sta
subplot(312),plot(Acceleration.Y)
subplot(313),plot(Acceleration.Z)


%% analyse data fft
X1 = dlmread('AccelerationX.txt'); 
X2 = dlmread('AccelerationY.txt');
X3 = dlmread('AccelerationZ.txt'); 
fs = 10;
N = 180000;
window = 5;
s_win = window*fs;



freX = fft(X1,180000);
freX_2 = abs(freX);
freX_1 = freX_2(1:N/2+1);
freX_1(2:end-1)= 2*freX_1(2:end-1);
f_freX = fs*(0:(N/2))/N;

% set up frequency signal
% set up size window
N = length(X1);
number_of_windows = floor(N/s_win);

% % Fourier Transform
% for i =1:number_of_windows % there are 700 windows 
%     % generating the frequency domain 
%     a = fft(X1((1+(i-1)*s_win):(i*s_win))); 
%     freX_2 = abs(a/s_win);  
%     freX_1 = freX_2(1:s_win/2+1);  
%     % extracting the frequency domain features 
%   meanfreX(i) = meanfreq(freX_1(1+(i-1)*s_win):(i*s_win)); % mean frequency
%    peakX(i) = findpeaks(freX_1(1+(i-1)*s_win):(i*s_win)); % peak frequency
%    medfreX(i) = medfreq(freX_1(1+(i-1)*s_win):(i*s_win)); % median frequency
%    powX(i) = ((abs(freX_1(1+(i-1)*s_win):(i*s_win))).^2)/s_win; %power spectrum
% end
%%

freX = fft(AX);
plot(abs(freX))

%%

F = fft(AX);
pow = F.*conj(F);
total_pow = sum(pow);

%% use tic tok later to time classification and testing time

