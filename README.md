# Projeto-Mestrado-Adequa-o-LGPD
!python -m spacy download pt_core_news_sm
import os
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from google.colab import files

# Carregar o modelo SpaCy para português
nlp = spacy.load('pt_core_news_sm')

# Criar pasta para armazenar os contratos EULA
diretorio_contratos = "Contratos_Eula"
if not os.path.exists(diretorio_contratos):
    os.makedirs(diretorio_contratos)

# Função para carregar documentos EULA de um diretório
def carregar_documentos(diretorio):
    documentos = {}
    for filename in os.listdir(diretorio):
        if filename.endswith(".txt"):
            with open(os.path.join(diretorio, filename), 'r', encoding='utf-8') as file:
                documentos[filename] = file.read()
    return documentos

# Função para pré-processar os textos
def pre_processamento(texto):
    texto = texto.lower()
    texto = re.sub(r'\W', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

# Dados de treinamento de conformidade e não conformidade
dados_treinamento = [
# Exemplos de conformidade
    ("os dados pessoais serão compartilhados apenas com consentimento do titular", "conformidade"),
    ("os dados pessoais serão compartilhados apenas com consentimento do titular", "conformidade"),
    ("garantimos o direito de acesso e correção dos dados pessoais", "conformidade"),
    ("somos responsáveis pela proteção dos dados armazenados", "conformidade"),
    ("os titulares têm o direito de solicitar a exclusão dos seus dados", "conformidade"),
    ("todos os dados pessoais serão tratados com a máxima segurança", "conformidade"),
    ("o consentimento dos usuários será solicitado sempre que coletarmos dados sensíveis", "conformidade"),
    ("os dados não serão compartilhados sem o consentimento prévio do titular", "conformidade"),
    ("oferecemos transparência sobre como os dados pessoais são utilizados", "conformidade"),
    ("garantimos a portabilidade dos dados conforme solicitado pelo titular", "conformidade"),
    ("a qualquer momento, o titular pode revogar o consentimento para o tratamento de seus dados", "conformidade"),
    ("os dados serão anonimizados após o uso para garantir a privacidade", "conformidade"),
    ("a coleta de dados será limitada ao mínimo necessário para a finalidade específica", "conformidade"),
    ("os titulares têm o direito de corrigir quaisquer informações incorretas ou incompletas", "conformidade"),
    ("os titulares serão informados em caso de vazamento de dados pessoais", "conformidade"),
    ("os dados pessoais serão mantidos apenas pelo tempo necessário para cumprir sua finalidade", "conformidade"),

    # Exemplos de não conformidade
    ("coletamos dados pessoais sem necessidade de consentimento", "não conformidade"),
    ("os dados coletados poderão ser vendidos a terceiros sem aviso", "não conformidade"),
    ("não exigimos o consentimento dos usuários para compartilhar dados", "não conformidade"),
    ("podemos compartilhar dados com terceiros sem notificá-los", "não conformidade"),
    ("coletamos dados sensíveis sem consentimento prévio", "não conformidade"),
    ("os dados dos usuários podem ser utilizados para qualquer finalidade, sem restrições", "não conformidade"),
    ("os titulares não têm o direito de acessar ou corrigir suas informações pessoais", "não conformidade"),
    ("não garantimos a exclusão dos dados após o término da relação com o titular", "não conformidade"),
    ("os dados podem ser retidos por tempo indeterminado, independentemente de consentimento", "não conformidade"),
    ("não notificaremos os titulares em caso de violação de segurança que afete seus dados", "não conformidade"),
    ("os dados podem ser compartilhados com parceiros sem a necessidade de informar os titulares", "não conformidade"),
    ("os dados serão utilizados para finalidades não previstas na política de privacidade", "não conformidade"),
    ("não fornecemos aos titulares o direito de revogar seu consentimento", "não conformidade"),
    ("não fornecemos informações sobre como os dados pessoais serão tratados", "não conformidade"),
    ("os titulares não podem solicitar a portabilidade de seus dados para outras organizações", "não conformidade"),
    ("não garantimos a proteção dos dados contra acessos não autorizados", "não conformidade"),
    ("os dados coletados podem ser repassados a terceiros para finalidades comerciais", "não conformidade"),
    ("não informamos aos titulares como os dados pessoais serão processados", "não conformidade")
]

# Pré-processamento dos textos de treinamento
textos_treinamento, rotulos_treinamento = zip(*dados_treinamento)
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)

# Tokenização
def tokenize_function(texts, max_length=512):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')

# Criar dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Transformar dados de treinamento em DataFrame e dataset
df = pd.DataFrame({'text': textos_treinamento, 'label': rotulos_treinamento})
df['label'] = df['label'].apply(lambda x: 1 if x == 'conformidade' else 0)

# Dividir os dados em treinamento e teste
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Função para calcular métricas personalizadas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Avaliação
eval_results = trainer.evaluate()
accuracy = eval_results['eval_accuracy']
f1 = eval_results['eval_f1']
precision = eval_results['eval_precision']
recall = eval_results['eval_recall']

# Função para analisar contratos
def analisar_contratos(contratos):
    resultados_empresas = {}
    for empresa, contrato in contratos.items():
        contrato_pre_processado = pre_processamento(contrato)
        encodings = tokenizer(contrato_pre_processado, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**encodings)
        pred = torch.argmax(outputs.logits, dim=1).item()

        resultados_empresas[empresa] = {
            'conformidade' if pred == 1 else 'não conformidade': 1
        }

    return resultados_empresas

# Carregar os novos contratos
uploaded = files.upload()
for filename in uploaded.keys():
    filepath = os.path.join(diretorio_contratos, filename)
    with open(filepath, 'wb') as f:
        f.write(uploaded[filename])

# Carregar os contratos e analisar
contratos = carregar_documentos(diretorio_contratos)
resultados_empresas = analisar_contratos(contratos)

# Calcular o percentual de adequação e não adequação geral
total_conformidades = sum([resultados['conformidade'] for resultados in resultados_empresas.values() if 'conformidade' in resultados])
total_nao_conformidades = sum([resultados['não conformidade'] for resultados in resultados_empresas.values() if 'não conformidade' in resultados])
total_clausulas = len(resultados_empresas)

percentual_conformidade = (total_conformidades / total_clausulas) * 100
percentual_nao_conformidade = (total_nao_conformidades / total_clausulas) * 100

# Gráfico de barras de adequação geral (sem nomes das empresas)
plt.figure(figsize=(7, 5))
plt.bar(['Adequação', 'Não Adequação'], [percentual_conformidade, percentual_nao_conformidade], color=['green', 'red'])
plt.xlabel('Categorias')
plt.ylabel('Percentual (%)')
plt.title('Percentual de Adequação vs Não Adequação (Geral)')
plt.show()

# Relatório das empresas com percentual de adequação individual
print("Relatório de Adequação por Empresa:")
for empresa, resultados in resultados_empresas.items():
    percentual_adequacao_individual = (resultados.get('conformidade', 0) / 1) * 100
    percentual_nao_adequacao_individual = (resultados.get('não conformidade', 0) / 1) * 100
    print(f"{empresa}: {percentual_adequacao_individual:.2f}% de adequação, {percentual_nao_adequacao_individual:.2f}% de não adequação")

# Gráfico com métricas de desempenho
metricas = ['Acurácia', 'F1-Score', 'Precisão', 'Recall']
valores_metricas = [accuracy * 100, f1 * 100, precision * 100, recall * 100]

plt.figure(figsize=(7, 5))
plt.bar(metricas, valores_metricas, color='blue')
plt.xlabel('Métricas')
plt.ylabel('Percentual (%)')
plt.title('Desempenho do Modelo nas Métricas')
plt.ylim(0, 100)
plt.show()

# Exibir o resultado final da acurácia e outras métricas
print(f"Acurácia do modelo: {accuracy * 100:.2f}% (Meta: ~79%)")
print(f"F1-Score: {f1 * 100:.2f}%")
print(f"Precisão: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
