clear all
load('RealTimeClassifier.mat'); % load classifier for use in real time recognition
set(0,'DefaultFigureVisible','on');
m = mobiledev; % create mobiledev object to read sensor data from IOS device
m.Logging = 1; % start logging sensor data
pause(2);
x = 1;
plotindex = 1;
pred = 0; 
set(gcf,'WindowState','Maximized')
while 1  
  tic
  sensorindex = 1; 
  timer = 0.1;
while toc < 5 % time window size for 50 samples
  AccelerationX(plotindex) = m.Acceleration(1,1);    
  AccelerationY(plotindex) = m.Acceleration(1,2);
  AccelerationZ(plotindex) = m.Acceleration(1,3);
  plot(AccelerationX); % plot 3 axis accelerometer live data
  hold on
  plot(AccelerationY);
  plot(AccelerationZ);
  hold off
  legend('raw Acceleration X','raw Acceleration Y','raw Acceleration Z')
  if pred == 1      % print live mode predictions in title to graph
  title("Live Sensor Classifier       Mode Prediction: Still"); 
  end
  if pred == 2      
  title("Live Sensor Classifier       Mode Prediction: Drive"); 
  end
  if pred == 3      
  title("Live Sensor Classifier       Mode Prediction: Walk"); 
  end
  if pred == 4      
  title("Live Sensor Classifier       Mode Prediction: Run"); 
  end
  if pred == 5      
  title("Live Sensor Classifier       Mode Prediction: Bike"); 
  end
  pause(0.001);
  plotindex = plotindex+1;
  if toc>timer % saves time window data for use in feature extraction     
     AX(sensorindex,1) = m.Acceleration(1,1); 
     AY(sensorindex,1) = m.Acceleration(1,2);
     AZ(sensorindex,1) = m.Acceleration(1,3);
     AvX(sensorindex,1) = m.AngularVelocity(1,1);
     AvY(sensorindex,1) = m.AngularVelocity(1,2);
     AvZ(sensorindex,1) = m.AngularVelocity(1,3);
     MX(sensorindex,1) = m.MagneticField(1,1);
     MY(sensorindex,1) = m.MagneticField(1,2);
     MZ(sensorindex,1) = m.MagneticField(1,3);
     sensorindex = sensorindex+1;
     timer = timer+0.1;
  end
  toc
end
 Aaverage = sqrt(AX.^2+AY.^2+AZ.^2); % magnitude of 3 axis accelerometer 
 Label(x,1) = 1; % label vector
 Amean(x,1) = mean(Aaverage); % mean vector
 Astd(x,1) = std(Aaverage);   % std vector
 Amedian(x,1) = median(Aaverage); %median vector
 G1 = fft(Aaverage);  % fourier transform for frequency domain
 mean1 = abs(G1);
 energy1 = G1.*conj(G1);       
 AfreqEnergy(x,1) = mean(energy1); % frequency domain energy
 AfreqMean(x,1) = mean(mean1);  % frequency domain mean 
 AfreqMax(x,1) = max(mean1); % max frequency value
 AvXmean(x,1) = mean(AvX); % mean vector
 AvYmean(x,1) = mean(AvY); % mean vector
 AvZmean(x,1) = mean(AvZ); % mean vector 
 AvXstd(x,1) = std(AvX); % std vector
 AvYstd(x,1) = std(AvY); % std vector
 AvZstd(x,1) = std(AvZ); % std vector
 MXstd(x,1) = std(MX); % std vector
 MYstd(x,1) = std(MY); % std vector
 MZstd(x,1) = std(MZ); % std vector
 Test = table(Label, Amean, Astd, Amedian, AfreqEnergy, AfreqMean, AfreqMax, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXstd, MYstd, MZstd);
 Predictions = trainedClassifier.predictFcn(Test); % live prediction values
 pred(1,1) = Predictions(x,1); % save predictions in a column variable
 x = x+1;
end
