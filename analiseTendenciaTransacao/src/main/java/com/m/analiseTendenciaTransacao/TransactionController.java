package com.m.analiseTendenciaTransacao;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@RestController // Anotação que indica que esta classe é um controlador REST, tratando de requisições HTTP.
public class TransactionController {

    @Autowired // Injeção de dependência do Spring para instanciar automaticamente o objeto 'network'.
    private MultiLayerNetwork network;

    @PostMapping("/analyze") // Anotação que indica que este método trata de requisições POST para o caminho "/analyze".
    public ResponseEntity<?> analyzeTransactions(@RequestBody List<Transaction> transactions) {
        // Preprocessa e cria a entrada para a rede neural
        INDArray features = Nd4j.create(transactions.size(), 3); // Cria um array para armazenar as características de entrada.
        for (int i = 0; i < transactions.size(); i++) {
            Transaction t = transactions.get(i);
            double establishmentCodeScaled = t.getEstablishmentCode() / 1000.0; // Normalização do código do estabelecimento.
            double amountScaled = t.getAmount() / 1000.0; // Normalização do montante da transação.
            double dayOfMonth = t.getTransactionDate().getDayOfMonth() / 31.0; // Normalização do dia do mês.
            features.putScalar(new int[]{i, 0}, establishmentCodeScaled);
            features.putScalar(new int[]{i, 1}, amountScaled);
            features.putScalar(new int[]{i, 2}, dayOfMonth);
        }

        // Realiza a inferência usando a rede neural
        INDArray output = network.output(features); // Obtem a saída da rede neural para as características fornecidas.
        double fraudThreshold = 0.5; // Define o limiar para considerar uma transação como fraude.

        // Gera os resultados formatados
        List<String> results = IntStream.range(0, transactions.size())
                .mapToObj(i -> String.format("Transaction ID %d: %s",
                        transactions.get(i).getEstablishmentCode(),
                        output.getDouble(i, 1) > fraudThreshold ? "is likely a fraud." : "is likely not a fraud."))
                .collect(Collectors.toList()); // Formata a saída para uma lista de strings baseada na probabilidade de fraude.

        return ResponseEntity.ok(results); // Retorna a lista de resultados como uma resposta HTTP.
    }
}
