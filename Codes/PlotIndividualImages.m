% This M-file constructs the individual images for 60 digits
% and plots them to a file.

clear
format short g
load ZipDigits.train%zip.train
digits=ZipDigits(:,1);
grayscale=ZipDigits(:,2:end);

[n,d]=size(grayscale);
w=floor(sqrt(d));

for i=1:1
	[i, digits(i)]
	curimage=reshape(grayscale(i,:),w,w);
	curimage=curimage';
	l=displayimage(curimage);
	sstr=['IndividualImages/image',int2str(i)];
%	eval(['print -deps ',sstr]);
end
