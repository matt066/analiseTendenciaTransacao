package com.m.analiseTendenciaTransacao;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@RestController
@Slf4j
public class TransactionController {

    @Autowired
    private ModelService modelService;

    @PostMapping("/analyze")
    public ResponseEntity<?> analyzeTransactions(@RequestBody List<Transaction> transactions) {
        MultiLayerNetwork model = modelService.getModel();
        if (model == null) {
            return ResponseEntity.status(500).body("Modelo não carregado");
        }

        // Preprocessa e cria a entrada para a rede neural
        INDArray features = Nd4j.create(transactions.size(), 3);
        for (int i = 0; i < transactions.size(); i++) {
            Transaction t = transactions.get(i);

            // Normalização dos dados da transação
            double establishmentCodeScaled = t.getEstablishmentCode() / 10000.0;
            double amountScaled = t.getAmount() / 1000000.0;
            double dayOfMonth = t.getTransactionDate().getDayOfMonth() / 31.0;

            features.putScalar(new int[]{i, 0}, establishmentCodeScaled);
            features.putScalar(new int[]{i, 1}, amountScaled);
            features.putScalar(new int[]{i, 2}, dayOfMonth);
        }

        // Realiza a inferência usando a rede neural
        INDArray output = model.output(features);
        double fraudThreshold = 0.7;

        // Map para agrupar as transações por establishmentCode
        Map<Integer, List<Map<String, Object>>> groupedResults = new HashMap<>();

        // Gera os resultados e agrupa por establishmentCode
        for (int i = 0; i < transactions.size(); i++) {
            Transaction t = transactions.get(i);
            int establishmentCode = t.getEstablishmentCode();
            boolean isFraud = output.getDouble(i, 1) > fraudThreshold;

            // Adiciona as informações da transação
            Map<String, Object> transactionInfo = new HashMap<>();
            transactionInfo.put("amount", t.getAmount());
            transactionInfo.put("transactionDate", t.getTransactionDate().toString());
            transactionInfo.put("isFraud", isFraud ? "Yes" : "No");

            // Agrupa as transações pelo establishmentCode
            groupedResults.computeIfAbsent(establishmentCode, k -> new ArrayList<>()).add(transactionInfo);
        }

        // Constrói o formato final da resposta
        StringBuilder formattedResponse = new StringBuilder();
        groupedResults.forEach((establishmentCode, transactionsList) -> {
            formattedResponse.append("EstablishmentCode ").append(establishmentCode).append(": {\n");
            for (Map<String, Object> transaction : transactionsList) {
                formattedResponse.append("  amount: ").append(transaction.get("amount"))
                        .append(", transactionDate: \"").append(transaction.get("transactionDate")).append("\"")
                        .append(", isFraud: \"").append(transaction.get("isFraud")).append("\"\n");
            }
            formattedResponse.append("}\n");
        });

        return ResponseEntity.ok(formattedResponse.toString());
    }
}
