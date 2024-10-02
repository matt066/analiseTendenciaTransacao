package com.m.analiseTendenciaTransacao;

import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;
import java.util.DoubleSummaryStatistics;
import java.util.stream.Collectors;


@Service
public class TransactionService {
    private final MultiLayerNetwork model;


    private static final double THRESHOLD_MULTIPLIER = 3.0;

    public TransactionService(MultiLayerNetwork model) {
        this.model = model;
    }

    public String analyzeTransactions(List<Transaction> transactions) {
            DoubleSummaryStatistics stats = transactions.stream()
                    .mapToDouble(Transaction::getAmount)
                    .summaryStatistics();
            double mean = stats.getAverage();
            double stddev = Math.sqrt(transactions.stream()
                    .mapToDouble(t -> Math.pow(t.getAmount() - mean, 2))
                    .sum() / stats.getCount());
            double threshold = mean + THRESHOLD_MULTIPLIER * stddev;

            return transactions.stream()
                    .map(t -> "Transaction ID " + t.getEstablishmentCode() +
                            (t.getAmount() > threshold ? " is anomalous." : " is normal."))
                    .collect(Collectors.joining("\n"));
    }


    public boolean isTransactionAnomalous(Double transactionAmount) {
        INDArray features = Nd4j.create(new double[]{transactionAmount}, 1, 1);
        INDArray output = model.output(features);
        return output.getDouble(1) > output.getDouble(0); // Probabilidade de anomalia maior que normal
    }
}

