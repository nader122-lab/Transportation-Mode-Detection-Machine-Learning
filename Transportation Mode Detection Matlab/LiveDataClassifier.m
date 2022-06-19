clear all
load('LiveClassifier.mat');
set(0,'DefaultFigureVisible','on');
m = mobiledev;
m.Logging = 1;
pause(2);
y = 1;
v = 1;
pred = 0; 
set(gcf,'WindowState','Maximized')
while 1  
  tic
  z = 1; 
  x = 0.1;
while toc < 5 
  a(v) = m.Acceleration(1,1);    
  b(v) = m.Acceleration(1,2);
  c(v) = m.Acceleration(1,3);
  plot(a);
  hold on
  plot(b);
  plot(c);
  hold off
  legend('raw Acceleration X','raw Acceleration Y','raw Acceleration Z')
  title("Live Sensor Classifier       Mode Prediction: " + pred);   
  pause(0.001);
  v = v+1;
  if toc>x     
     AX(z,1) = m.Acceleration(1,1);
     AY(z,1) = m.Acceleration(1,2);
     AZ(z,1) = m.Acceleration(1,3);
     AvX(z,1) = m.AngularVelocity(1,1);
     AvY(z,1) = m.AngularVelocity(1,2);
     AvZ(z,1) = m.AngularVelocity(1,3);
     MX(z,1) = m.MagneticField(1,1);
     MY(z,1) = m.MagneticField(1,2);
     MZ(z,1) = m.MagneticField(1,3);
     z = z+1;
     x = x+0.1;
  end
  toc
end
 Label(y,1) = 1;
 AXmean(y,1) = mean(AX);
 AYmean(y,1) = mean(AY); 
 AZmean(y,1) = mean(AZ)'; 
 AXstd(y,1) = std(AX); 
 AYstd(y,1)  = std(AY); 
 AZstd(y,1) = std(AZ); 
 AXmedian(y,1) = median(AX);
 AYmedian(y,1) = median(AY);
 AZmedian(y,1) = median(AZ);
 G1 = fft(AX);
 mean1 = abs(G1);
 pow1 = G1.*conj(G1);
 G2 = fft(AY);
 mean2 = abs(G2);
 pow2 = G2.*conj(G2);
 G3 = fft(AZ);
 mean3 = abs(G3);
 pow3 = G3.*conj(G3);        
 AXfreqEnergy(y,1) = mean(pow1);
 AYfreqEnergy(y,1) = mean(pow2);
 AZfreqEnergy(y,1) = mean(pow3);
 AXfreqMean(y,1) = mean(mean1); 
 AYfreqMean(y,1) = mean(mean2);
 AZfreqMean(y,1) = mean(mean3);
 AvXmean(y,1) = mean(AvX);
 AvYmean(y,1) = mean(AvY);
 AvZmean(y,1) = mean(AvZ); 
 AvXstd(y,1) = std(AvX);
 AvYstd(y,1)  = std(AvY);
 AvZstd(y,1) = std(AvZ);
 MXmean(y,1) = mean(MX);
 MYmean(y,1) = mean(MY);
 MZmean(y,1) = mean(MZ);
 MXstd(y,1) = std(MX);
 MYstd(y,1)  = std(MY);
 MZstd(y,1) = std(MZ);
 Test = table(Label, AXmean, AYmean, AZmean, AXstd, AYstd, AZstd, AXmedian, AYmedian, AZmedian, AXfreqEnergy, AYfreqEnergy, AZfreqEnergy, AXfreqMean, AYfreqMean, AZfreqMean, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXmean, MYmean, MZmean, MXstd, MYstd, MZstd);
 Predictions = trainedClassifier.predictFcn(Test);
 pred(1,1) = Predictions(y,1);
 y = y+1;
end
