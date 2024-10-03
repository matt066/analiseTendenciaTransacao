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
        INDArray features = Nd4j.create(transactions.size(), 4); // 4 colunas agora
        for (int i = 0; i < transactions.size(); i++) {
            Transaction t = transactions.get(i);

            // Normalização dos dados da transação
            double establishmentCodeScaled = t.getEstablishmentCode() / 10000.0; // Normalização arbitrária do establishmentCode
            double amountScaled = t.getAmount() / 1000000.0;  // Normalização do valor da transação
            double dayOfMonth = t.getTransactionDate().getDayOfMonth() / 31.0;  // Normalização do dia do mês
            double avgTransactionValueScaled = t.getAverageTransactionValue() / 1000000.0;  // Normalização do valor médio de transações

            // Popula o array de entrada para o modelo
            features.putScalar(new int[]{i, 0}, establishmentCodeScaled);
            features.putScalar(new int[]{i, 1}, amountScaled);
            features.putScalar(new int[]{i, 2}, dayOfMonth);
            features.putScalar(new int[]{i, 3}, avgTransactionValueScaled);  // O novo input

        }

        // Realiza a inferência usando o modelo
        INDArray output = model.output(features);
        double fraudThreshold = 0.8;

        // Agrupando resultados por establishmentCode
        Map<Integer, List<Map<String, Object>>> groupedResults = new HashMap<>();
        for (int i = 0; i < transactions.size(); i++) {
            Transaction t = transactions.get(i);
            int establishmentCode = t.getEstablishmentCode();
            double fraudScore = output.getDouble(i, 1);  // Pega o score de fraude

            // Adiciona as informações da transação
            Map<String, Object> transactionInfo = new HashMap<>();
            transactionInfo.put("amount", t.getAmount());
            transactionInfo.put("transactionDate", t.getTransactionDate().toString());
            transactionInfo.put("averageTransactionValue", t.getAverageTransactionValue()); // Adiciona o ticket médio
            transactionInfo.put("fraudScore", fraudScore);

            // Agrupa as transações por establishmentCode
            groupedResults.computeIfAbsent(establishmentCode, k -> new ArrayList<>()).add(transactionInfo);
        }

        // Constrói a resposta final
        StringBuilder formattedResponse = new StringBuilder();
        groupedResults.forEach((establishmentCode, transactionsList) -> {
            formattedResponse.append("EstablishmentCode ").append(establishmentCode).append(": {\n");
            for (Map<String, Object> transaction : transactionsList) {
                formattedResponse.append("  amount: ").append(transaction.get("amount"))
                        .append(", transactionDate: \"").append(transaction.get("transactionDate")).append("\"")
                        .append(", averageTransactionValue: ").append(transaction.get("averageTransactionValue"))
                        .append(", fraudScore: \"").append(transaction.get("fraudScore")).append("\"\n");
            }
            formattedResponse.append("}\n");
        });

        return ResponseEntity.ok(formattedResponse.toString());
    }

}
