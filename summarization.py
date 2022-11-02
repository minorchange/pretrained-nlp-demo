import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
from long_text import wiki_elefant_text


def generate_summary(text):
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ckpt = 'mrm8488/bert2bert_shared-german-finetuned-summarization'
   tokenizer = BertTokenizerFast.from_pretrained(ckpt)
   model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
   inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids.to(device)
   attention_mask = inputs.attention_mask.to(device)
   output = model.generate(input_ids, attention_mask=attention_mask)
   return tokenizer.decode(output[0], skip_special_tokens=True)
   
# text = wiki_elefant_text
# text = """In der menschlichen Gesellschaftsentwicklung und Geschichte spielten Elefanten eine bedeutende Rolle. Sie wurden anfänglich als Nahrungsressource und Rohstoffquelle gejagt beziehungsweise genutzt, fanden bereits vor mehr als 30.000 Jahren Einzug in Kunst und Kultur und erlangten in späterer Zeit mit der Sesshaftwerdung und der Entstehung verschiedener Hochkulturen ebenfalls große Bedeutung. Einzig der Asiatische Elefant trat als gezähmtes Tier dauerhaft in den Dienst des Menschen. Er fungierte dabei zunächst als Last- und Arbeitstier, später wurde er in Kriegen eingesetzt und galt als Zeichen außerordentlicher Größe und Macht. Die wissenschaftliche Erstbeschreibung des Afrikanischen und Asiatischen Elefanten datiert in das Jahr 1758. Beide Arten wurden zunächst einer einzigen Gattung zugewiesen, erst Anfang des 19. Jahrhunderts erfolgte die generische Trennung der beiden Vertreter. Der Waldelefant ist erst seit dem Beginn der 2000er Jahre als eigenständige Art anerkannt. Die Familie der Elefanten wurde 1821 eingeführt. Die Bestände der drei Arten gelten in unterschiedlichem Maße als gefährdet."""
# s = generate_summary(text)
# print(s)