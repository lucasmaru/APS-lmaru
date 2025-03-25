clc
clear all

%pkg install -forge control % esto instala el paquete de control, en teoria solo se ahce una vez
%pkg load control %Sin embargo si debemos cargar el paquete cada vez, conviene hacerlo en el command

Wo=1;
Q=5;
%Q=sqrt(2)/2;
%H = tf ( [ Wo/Q 0 ] , [1 Wo/Q (Wo)**2] );
H = tf ( [ 1 0 0 ] , [1 Wo/Q (Wo)**2] );
bode(H);

