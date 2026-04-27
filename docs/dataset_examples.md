# Dataset examples

Random samples (seed=42) from each interim file after preprocessing.
These are the records that will be fed into merge_and_split → train/valid/test.

## IBM ArgQ 30K — real debate pairs (EN)

- File: `data/interim/en_ibm_argq.jsonl`
- Total records: 29,664

### Example 1

- **Topic**: Intelligence tests bring more harm than good
- **Input**: some people just don't test well and scoring badly on intelligence tests can have profound impacts that are not deserved.
- **Target**: the intelligence tests allow knowing the abilities of the people that help them to take an education according to their possibilities
- **Meta**: `input_stance`=pro · `target_stance`=con · `is_synthetic`=False · `quality_input_WA`=1.0 · `quality_target_WA`=1.0

### Example 2

- **Topic**: We should ban cosmetic surgery for minors
- **Input**: We should ban cosmetic surgery for minors because parents will pressure their children into having permanent surgical modifications.
- **Target**: you need to look at the circumstances. what if a child was in an accident and need cosmetic surgery in order to fix themselves and lead a normal life?
- **Meta**: `input_stance`=pro · `target_stance`=con · `is_synthetic`=False · `quality_input_WA`=0.912402945 · `quality_target_WA`=0.837043071

### Example 3

- **Topic**: Assisted suicide should be a criminal offence
- **Input**: assisted suicide should be allowed because it allows people with terminal illnesses to die with dignity before suffering a long and painful death.
- **Target**: Assisted suicide should be a criminal offence as if the person cannot make their own decisions and actions then it would be impossible to know if the person helping wasn't just murdering them.
- **Meta**: `input_stance`=con · `target_stance`=pro · `is_synthetic`=False · `quality_input_WA`=1.0 · `quality_target_WA`=1.0

### Example 4

- **Topic**: We should prohibit women in combat
- **Input**: if they can pass all training and everything what`s need to get involve in the combat, can`t see why they should not be in the combat?
- **Target**: this may sound old fashioned but men should be the protectors
- **Meta**: `input_stance`=con · `target_stance`=pro · `is_synthetic`=False · `quality_input_WA`=0.526910465 · `quality_target_WA`=0.244689796

### Example 5

- **Topic**: We should ban telemarketing
- **Input**: telemarketing is invasive and upsets many people so it should be banned.
- **Target**: we shouldn't ban telemarketing because it helps companies obtain more business
- **Meta**: `input_stance`=pro · `target_stance`=con · `is_synthetic`=False · `quality_input_WA`=0.864556376 · `quality_target_WA`=0.83891061

## mc-ai/conversation_dataset — persuasion-only after filter (EN)

- File: `data/interim/en_mc_conversation.jsonl`
- Total records: 1,500

### Example 1

- **Topic**: Charity Donation Discussion
- **Input**: Ok i'm happy to at least get something. So out of your two dollar payment you are only willing to donate 10 cents?
- **Target**: I am not"only" willing. I am willing to donate 10 cents.
- **Meta**: `is_synthetic`=False · `conversation_id`=1195 · `corpus_id`=persuasionforgood

### Example 2

- **Topic**: Charity and Donations
- **Input**: Yes, they are both good charities. It has been very nice chatting with you.
- **Target**: Yes, nice chatting with you too!
- **Meta**: `is_synthetic`=False · `conversation_id`=8509 · `corpus_id`=persuasionforgood

### Example 3

- **Topic**: Discussion about donations to a children's charity and personal perspectives on family and parenting.
- **Input**: Whatever amount you choose to donate from your payment for this taks will be donated to a children's cahrtiy called save the children. Save the children is an international governmental organization that promotes children's rights, provides relief and hleps support children in developing countries.
- **Target**: I see. I guess i can donate $0.20 for now, as i am not guaranteed of how much i will be paid for this task. I really want to support save the children since i'm expecting my first child this december And i believe children are a gift from god and they should be cherished and taken care of.
- **Meta**: `is_synthetic`=False · `conversation_id`=6197 · `corpus_id`=persuasionforgood

### Example 4

- **Topic**: Discussion about donating to the charity Save the Children and general conversation afterwards.
- **Input**: Thank you for your time, and for the opportunity.
- **Target**: You're very welcome! Thank you for chatting with me.
- **Meta**: `is_synthetic`=False · `conversation_id`=11465 · `corpus_id`=persuasionforgood

### Example 5

- **Topic**: Charitable donations and organization information
- **Input**: Your donation will be directly deducted from your task payment.
- **Target**: Ok cool, weve got a few more, any other charities
- **Meta**: `is_synthetic`=False · `conversation_id`=9145 · `corpus_id`=persuasionforgood

## Isotonic/human_assistant_conversation — markers and code filtered (EN)

- File: `data/interim/en_isotonic_conversation.jsonl`
- Total records: 1,500

### Example 1

- **Topic**: casual conversation
- **Input**: In this task, you will be shown an incorrect English sentence. You need to generate a corrected form of the input sentence. Input: Also , most of people prefer to entertain athletics rather than other arts .
- **Target**: Also , most people prefer athletics to other arts .
- **Meta**: `is_synthetic`=False

### Example 2

- **Topic**: casual conversation
- **Input**: Task: How many ways can i arrange 3 books and 2 pens?
- **Target**: There are 3! = 6 ways.
- **Meta**: `is_synthetic`=False

### Example 3

- **Topic**: casual conversation
- **Input**: In this task, you're given a four sentences of story written in natural language. Your job is to complete end part of the story by predicting appropriate last sentence which is coherent with the given sentences. Input: Sentence1: Danny was on a trip with his family. Sentence2: They had just arrived at a hotel for the night. Sentence3: Danny wanted to put his pajamas on but he couldn't find his bag. Sentence4: They realized that his bag hadn't made it into the car.
- **Target**: Danny's mom had to go buy him some clothes for the trip.
- **Meta**: `is_synthetic`=False

### Example 4

- **Topic**: casual conversation
- **Input**: Task: Write down 10 ways of saying "i'm sorry".
- **Target**: I'm sorry, I apologize, I regret, I feel bad about it, I wish I hadn't done that, I didn't mean to do that, I was wrong, I shouldn't have done that, I wish I could take it back, I wish I had done something different.
- **Meta**: `is_synthetic`=False

### Example 5

- **Topic**: casual conversation
- **Input**: How does currency exchange work.
- **Target**: Currency exchange is the process of converting one currency into another currency. It involves the exchange of one currency for another at an agreed rate of exchange. To carry out a currency exchange, you would need to go to a bank, foreign exchange dealer, or a licensed money changer. Here, the exchange rate would be determined and the respective currencies would be exchanged according to that.
- **Meta**: `is_synthetic`=False

## SohamGhadge/casual-conversation — chit-chat (EN)

- File: `data/interim/en_casual_conversation.jsonl`
- Total records: 1,000

### Example 1

- **Topic**: casual conversation
- **Input**: how do you like it so far?
- **Target**: i like it so far. my classes are pretty good right now.
- **Meta**: `is_synthetic`=False

### Example 2

- **Topic**: casual conversation
- **Input**: you can see the stars so much more clearly after it rains.
- **Target**: i would love for it to rain today.
- **Meta**: `is_synthetic`=False

### Example 3

- **Topic**: casual conversation
- **Input**: well, congratulations.
- **Target**: thank you.
- **Meta**: `is_synthetic`=False

### Example 4

- **Topic**: casual conversation
- **Input**: i'm serious. you deserved this promotion.
- **Target**: is that what you really think?
- **Meta**: `is_synthetic`=False

### Example 5

- **Topic**: casual conversation
- **Input**: i'm really glad that you came to see me.
- **Target**: i had to. i was missing you a lot.
- **Meta**: `is_synthetic`=False

## KLUE-NLI — contradiction pairs (KO, no vLLM)

- File: `data/interim/ko_klue_nli.jsonl`
- Total records: 5,000

### Example 1

- **Topic**: 두 문장의 관점 차이
- **Input**: 미국여행시 위치가 이보다 더 좋을순 없습니다.
- **Target**: 로마여행시 위치가 이보다 더 좋을순 없습니다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 2

- **Topic**: 두 문장의 관점 차이
- **Input**: 1인별 구매한도를 월 50만원에서 40만원 이내로 줄일 방침이다.
- **Target**: 1인별 구매한도를 월 50만원에서 100만원 이내로 늘리고 올해 잔여 물량 6조 5000억원을 차질없이 유통할 계획이다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 3

- **Topic**: 두 문장의 관점 차이
- **Input**: 그가 만들어준 상그리아는 별로에요.
- **Target**: 또한 그가 직접 만든 상그리아도 최고예요.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 4

- **Topic**: 두 문장의 관점 차이
- **Input**: 괴이한 내용과 허접한 구성 참 더러워 보였습니다
- **Target**: 내용은 괴이했지만 구성은 완벽했습니다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 5

- **Topic**: 두 문장의 관점 차이
- **Input**: 또 면회객은 손 소독을 반드시 하고 마스크와 비닐장갑을 착용하며, 기관에서는 발열체크 등 위험요인을 확인하고 출입명부를 비치해 관리해야 한다.
- **Target**: 면회객은 마스크만 착용하면 된다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train
