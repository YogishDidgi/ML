function a = hw2_E1_10()
    N = 6;
    mu = 0.5;
    i = 0:1:6;
    for k = 1:length(i)
x        P(k)=binomial(6,uint8(k))*(0.5^k)*(0.5^(6-k));
    end
    for k = 1:length(i)
        for j = 1:length(i)
            A(k,j) = P(k)*P(k);
        end
    end
    e = 0:0.1:1;
    for k = 1:length(e)
        B(k) = sum(abs((A/N)-mu)>e(k));
    end
    B = B/(length(i)*length(i));
    plot(e, B);
end

function freqDist = hw2_E1_10_part1(numCoins, numFlips, numExpts)
    close all;
    %Head - 1; Tail - 0
    xAxis = 0:numFlips;
    freqDist = zeros(numExpts, 3);
    for iter = 1:numExpts
        freqMinCoin = 0;
        freqMinValue = 11;
        coinOutcome = zeros(numCoins, 1);%outcome itself is not important. only the number of heads obtained
        for i = 1:numCoins
            for j = 1:numFlips
                if(rand() >= 0.5)
                    coinOutcome(i) = coinOutcome(i) + 1;
                end
            end
            if(freqMinValue > coinOutcome(i))
                freqMinValue = coinOutcome(i);
                freqMinCoin = i;
            end
        end
        freq1 = coinOutcome(1);
        freqRandCoin = int64(randi([1 numCoins]));
        freqRandValue = coinOutcome(freqRandCoin);
        freqDist(iter, 1) = freq1;
        freqDist(iter, 2) = freqRandValue;
        freqDist(iter, 3) = freqMinValue;
    end
    figure,
    subplot(3,1,1), hist(freqDist(:,1), xAxis);hold on;
    xlabel('Frequency');
    ylabel('Number of Heads');
    legend('Coin-1');
    subplot(3,1,2), hist(freqDist(:,2), xAxis);hold on;
    xlabel('Frequency');
    ylabel('Number of Heads');
    legend('Coin-Rand');
    subplot(3,1,3), hist(freqDist(:,3), xAxis);
    xlabel('Frequency');
    ylabel('Number of Heads');
    legend('Coin-Min');
    freqDistNormalized = freqDist/(numFlips);
    mu = 0.5;
    val1 = zeros(11, 1);
    val2 = zeros(11, 1);
    val3 = zeros(11, 1);
    eps = 0:0.1:1;
    for epsilon = 0:0.1:1
        prob1 = (abs(freqDistNormalized(:,1) - mu) > epsilon);
        prob2 = (abs(freqDistNormalized(:,2) - mu) > epsilon);
        prob3 = (abs(freqDistNormalized(:,3) - mu) > epsilon);
        total1 = 0;
        total2 = 0;
        total3 = 0;
        for i = 1:numExpts
            if(prob1(i) == 1)
                total1 = total1 + 1;%freqDistNormalized(i,1);
            end
            if(prob2(i) == 1)
                total2 = total2 + 1;%freqDistNormalized(i,2);
            end
            if(prob3(i) == 1)
                total3 = total3 + 1;%freqDistNormalized(i,3);
            end
        end
        total1 = total1/numExpts;
        total2 = total2/numExpts;
        total3 = total3/numExpts;
        val1(int64(epsilon*10 + 1)) = total1;
        val2(int64(epsilon*10 + 1)) = total2;
        val3(int64(epsilon*10 + 1)) = total3;
    end
    
    f=2*exp(-2*numFlips*eps.*eps);
    figure, plot(eps, val1, eps, val2, eps, val3, eps, f);
    xlabel('epsilon');
    ylabel('Probability');
    legend('Coin-1','Coin-Rand','Coin-Min','Hoeffding Bound');
    
    
end