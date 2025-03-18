# Projeto-Mestrado-Adequa-o-LGPD
Um Sistema Baseado em Processamento de Linguagem Natural (PLN) para Análise de Contratos de Aceite de Termos (EULA) para Adequação à LGPD 
fluxo tem como objetivo a análise automatizada de contratos de Termos de Uso (EULA) para garantir sua conformidade com a Lei Geral de Proteção de Dados (LGPD). 
Dividir base de dados – sintéticos / Real Verificando Métricas.

Fluxo do método, com os principais etapas e técnicas utilizados:

1. Coleta de Dados : A primeira etapa do projeto envolve a coleta de contratos EULA (End User License Agreements) que serão analisados. Esses contratos são obtidos a partir de diferentes fontes e constituem o dataset de análise, que serve como base de dados para o sistema de Processamento de Linguagem Natural (PLN). Foi inicialmente feita manual , sendo aplicada api para automatizar este projeto de seleção de documentos.

2. Inserção dos Contratos EULA : Nesta etapa, os contratos coletados são inseridos no sistema para serem submetidos a análise. Esta fase garante que os documentos sejam adequadamente carregados e formatados para o pré-processamento subsequente.

3. Pré-processamento : O pré-processamento dos contratos é para preparar os dados brutos para a análise de PLN. Aqui, são aplicadas técnicas como a tokenização,(BertTokenizer e BertForSequenceClassification) que consiste em dividir o texto em unidades menores, como palavras ou tokens, facilitando a análise subsequente pelo modelo de linguagem.

4. Uso de Transformers BERT : A etapa central da análise se dá pelo uso de Transformers BERT (Bidirectional Encoder Representations from Transformers). O BERT quem realiza uma análise profunda do contexto dos termos presentes nos contratos, sendo capaz de entender e interpretar as cláusulas jurídicas e identificar termos que possam não estar em conformidade com a LGPD.

5. Análise dos Contratos : Uma vez pré-processados, os contratos são submetidos à análise de conformidade. O sistema verifica se os contratos atendem aos requisitos da LGPD, comparando as cláusulas com um conjunto de regras e melhores práticas relacionadas à privacidade de dados.

6. Classificação de Adequação ou Não Adequação: O sistema utiliza o modelo BertForSequenceClassification, que faz a classificação dos contratos como adequados ou não adequados em relação à LGPD. Esta classificação é feita com base nas saídas geradas pelo modelo BERT.

7. Métricas de Desempenho: O desempenho do sistema de classificação é avaliado por meio de várias métricas: sklearn.metrics
-Acurácia: Mede a porcentagem de contratos corretamente classificados. : accuracy_score
-Precisão: Indica a proporção de contratos classificados como adequados que realmente são.:precision_score
-Recall: Mede a capacidade do sistema de identificar todos os contratos inadequados.: recall_score
-F1 score: Combina precisão e recall em uma única métrica harmônica. : f1_score

8. Relatório de Conformidade: Após a classificação, o sistema gera um relatório de conformidade para cada contrato analisado, detalhando os pontos em que ele está em conformidade ou fora da conformidade com a LGPD. Este relatório é essencial para fornecer feedback aos autores dos contratos sobre as áreas que precisam de ajustes.

9. Correção dos Contratos Não Conformes , para adequação ao LGPD feitas automaticamente, Gerando um novo diretorio com os contrato corrigidos.
Para os contratos classificados como não conformes, são sugeridas correções específicas. Essas correções são orientadas para ajustar as cláusulas problemáticas de modo que o contrato esteja em conformidade com as exigências da LGPD.

Este fluxo detalhado, baseado em técnicas avançadas de PLN, visa otimizar o processo de análise e garantir que as empresas cumpram os requisitos legais, economizando tempo e recursos na verificação manual de contratos.
![image](https://github.com/user-attachments/assets/72be0f69-5efa-4918-b9ed-55d0992240a9)
