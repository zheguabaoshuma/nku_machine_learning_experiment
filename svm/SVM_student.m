%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % ��������С
center1 = [1,1];        % ��һ����������
center2 = [6,6];        % �ڶ�����������
%���Կɷ����ݣ�center2 = [6,6]�����Բ��ɷ����ݣ���Ϊcenter2 = [3,3]
X = zeros(2*n,2);       % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(2*n,1);       % ����ǩ
X(1:n,:) = ones(n,1)*center1 + randn(n,2);
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %����X��ǰn�б�ʾ���1�����ݣ���n�б�ʾ���2������
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % ��һ�����ݱ�ǩΪ1���ڶ���Ϊ-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  SVMģ��   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ѧ��ʵ��,���SVM�Ĳ���(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % SVM: y = x*w + b
alpha = zeros(2*n,1);       % ��ż�������

%%%%%%%% %%%%%%%% ʹ�����������������շ�ѵ��ģ��


%%%%%%%%%%%%%%%%  ����������ͼ  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% ������ x*w + b =0 ��ͼ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % �������
                                                           % x1Ϊ���������ᣬy1Ϊ����
y2 = ( ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);
y3 = ( -ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);  %��������߽�

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % ���������
hold on;
plot( x1,y2,'k-.','LineWidth',1,'MarkerSize',10);                         % ���ּ���߽�
hold on;
plot( x1,y3,'k-.','LineWidth',1,'MarkerSize',10);                         % ���ּ���߽�
hold on;
plot(X(alpha>0,1),X(alpha>0,2),'rs','LineWidth',1,'MarkerSize',10);    % ��֧������
hold on;
plot(X(alpha<C&alpha>0,1),X(alpha<C&alpha>0,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);    % ������߽��ϵ�֧������
hold on;
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',10)
legend('class 1','class 2','classification surface','boundary','boundary','support vectors','support vectors on boundary');
