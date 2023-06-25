%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 2000;                % ��������С
X = rand(n,2)*10;        % n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(n,1);          % ����ǩ

for i=1:n
   if 0<X(i,1) && X(i,1)<3 && 0<X(i,2) && X(i,2)<3              % ����x��y������ȷ������      
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 2;
   end
   if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       Y(i) = 3;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 0<X(i,2) && X(i,2)<3
       Y(i) = 4;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 5;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 7<X(i,2) && X(i,2)<10
       Y(i) = 6;
   end
   if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       Y(i) = 7;
   end
   if 7<X(i,1) && X(i,1)<10 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 8;
   end;
   if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       Y(i) = 9;
   end;
end

X = X(Y>0,:);                                                    % ע��X����[0,10]*[0,10]��Χ�ھ������ɵģ�������ֻ�����һ����X�����֮��İ�ɫ����еĵ�û�б꣬�����Ҫ����Щ��ȥ��
Y = Y(Y>0,:);                                                    % X(Y>0,:)��ʾֻȡX�ж�Ӧ��Y����0���У�������Ϊ��ɫ����еĵ��Y��Ϊ0
nn = length(Y);                                                  % ȥ������ɫ���ʣ�µĵ�

n = 2000;
X(nn+1:n,:) = rand(n-nn,2)*10;                                   % ����n-nn��������
Y(nn+1:n) = ceil( rand(n-nn,1)*9 );                              % ������ı�ǩ���ѡȡ��rand(n-nn,1)*9��ʾ����[0,9]�ľ��ȷֲ���ceil��ʾ��ȡ�����ʽ��Ϊ1,2,...,9

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�    X(Y==1,1)��ʾ���Ϊ1��Y==1���ĵ�ĵ�һά�����꣬X(Y==1,2)��ʾ���Ϊ1�ĵ�ĵڶ�ά������
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==5,1),X(Y==5,2),'m*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');

%%%%%%%%%%%%%%%%%%%  ���ɲ�������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% ���ɲ�������:��ѵ������ͬ�ֲ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 100;                % ������������С
Xt = rand(m,2)*10;       
Yt = zeros(m,1);
for i=1:m
   if 0<Xt(i,1) && Xt(i,1)<3 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 1;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 2;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 3;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 4;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 5;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 6;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 7;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 8;
   end;
   if 7<Xt(i,1) && Xt(i,1)<10 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 9;
   end;
end
Xt = Xt(Yt>0,:);
Yt = Yt(Yt>0,:);
m = length(Yt);
Ym = zeros(m,1);                                                         % ��¼ģ��������

figure(2)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
plot(Xt(:,1),Xt(:,2),'ms','MarkerFaceColor','m','LineWidth',1,'MarkerSize',10);            % ���������ݵ�   Xt(:,2)��ʾXt�ĵڶ��У�����������
hold on;
xlabel('x axis');
ylabel('y axis');

%%%%%%%%%%%%%%%%%%  ��Ҷ˹�㷨��ѧ��ʵ��     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ����ģ�͵�Ԥ�����������������ݵ���ʵ����Ƚϣ����������     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%�����Խ�������Ym��

%%%%%%%%%%%%%%%%%%  ������ӻ�     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
plot(Xt(Ym==1,1),Xt(Ym==1,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(Xt(Ym==2,1),Xt(Ym==2,2),'ks','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(Xt(Ym==3,1),Xt(Ym==3,2),'bs','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==4,1),Xt(Ym==4,2),'gs','MarkerFaceColor','g','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==5,1),Xt(Ym==5,2),'bs','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==6,1),Xt(Ym==6,2),'cs','MarkerFaceColor','c','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==7,1),Xt(Ym==7,2),'bs','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==8,1),Xt(Ym==8,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(Xt(Ym==9,1),Xt(Ym==9,2),'ks','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');