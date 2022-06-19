  a(v) = m.Acceleration(1,1);    
  b(v) = m.Acceleration(1,2);
  c(v) = m.Acceleration(1,3);
  plot(a);
  hold on
  plot(b);
  plot(c);
  hold off
  legend('Acceleration X','Acceleration Y','Acceleration Z')
  if pred == actual
     title("Live Sensor Classifier       Mode Prediction: " + pred + "      Actual Mode: " + actual + "    Prediction: CORRECRT");
  else
     title("Live Sensor Classifier       Mode Prediction: " + pred + "      Actual Mode: " + actual + "    Prediction: INCORRECT");    
  end    
  pause(0.01);
  v = v+1;