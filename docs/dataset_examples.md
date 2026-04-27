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

## Korean Petitions — vLLM-synthesized rebuttals (KO, Qwen3-235B-A22B)

- File: `data/interim/ko_korean_petitions.jsonl`
- Total records: 9,992

Samples are stratified by petition category to show topic diversity (the
source corpus is dominated by 소년/청소년 법 petitions — those are filtered out
here for the showcase, but they remain in the training data).

### Example 1 — 안전/환경

- **Topic**: 범죄자 신상 공개법
- **Input**: 절도부터 살인까지 모든 범죄자의 신상 정보를 기록하고 해당 경위를 서술한 내용과 얼굴을 공개하는 법안을 발의해 주시기 바랍니다. 현재 우리나라는 죄지은 사람들 얼굴이 공개되지 않습니다. 인권 보호라는 이유 때문이죠. 이는 죄를 지어도 같은 사람이다라는 생각이 전제 되어 있기 때문이라 생각하는데 저는 이런 기본 전제가 지켜지기는 하되 솜방망이 처벌이라는 소리가 나오지 않도록 바뀌어야한다고 생각합니다.
- **Target**: 범죄자의 신상 공개가 범죄 예방보다는 사회적 낙인과 재범 방지에 오히려 악영향을 줄 수 있다는 점을 고려해야 합니다. 처벌의 강화보다는 범죄 원인 해소와 사회 복귀 지원이 더 근본적인 해결책이 될 수 있습니다.

### Example 2 — 육아/교육

- **Topic**: 보육료 현실화 , 반별인건비 지원
- **Input**: 안녕하세요 구로구에 있는 어린이집원장입니다 이번 최저임금 인상으로 어린이집 운영이 어려워짐에 따라 이렇거 글을 올립니다 근무하는 선생님들에게 더나은 근무환경과 처우를 해주어 의욕을 높이는것은 좋으나 보육료가 오르지 않은 상황에서 최저임금을 올리게되면 당장 원 운영이 심각하게 어려워 집니다 . 영아전문가인 보육교사가 대한민국 미래의 초석을 세운다는 신념을 갖고 우리 아이들과 행복하게 보내기에 전념을 해야하는 시기에 이렇게 운영의 어려움에 대한 걱정으로 중요한 시간을 놓치고있습니다 보육환경과 보육교사
- **Target**: 보육료가 오르지 않아 운영이 어려운 현실을 잘 알고 있지만, 보육교사의 안정적인 처우 없이는 양질의 보육 서비스가 어렵고, 이는 궁극적으로 아이들의 성장에 직접적인 영향을 줍니다.

### Example 3 — 정치개혁

- **Topic**: 소수의 의견은 전달 되지 않는 것인지요? 청원을 보시는 모든 분들께 동의르 구합니다.
- **Input**: 청원 화면에 '일정 수준 이상의 추천을 받고 국정 현안으로 분류된 청원에 대해, 가장 책임있는 정부 및 청와대 당국자(장관, 대통령 수석비서관 등)의 답변 을 받을 수 있습니다. '라고 명시 되어 있습니다. 일정이상의 추천이란? 구체적으로 얼마의 추천의 의미하는지 궁급합니다. 이는 해석하기에 따라 소규모 모임은 상대적으로 참여 인원이 적기 때문에, 기회의 평등으로부터 멀어질 수 있다고 생각 합니다. 저희 전국노인요양공동생활가정연합회의 인원은 200명 미만 입니다. 그렇다면.
- **Target**: 청원 시스템은 일정 수의 추천을 기준으로 답변 대상을 선정하지만, 이는 인원 수보다는 사회적 공감과 관심의 정도를 반영하려는 목적을 가지고 있습니다. 소규모 단체라도 그 청원의 내용이 타당하고 널리 공감받을 수 있다면 충분히 목소리를 전달할 기회가 있습니다.

### Example 4 — 외교/통일/국방

- **Topic**: [지원여성에 한하여 여성사병모집] 대한민국 '여성병장'이 되고 싶습니다.
- **Input**: 청원내용 : □ 희망 및 지원여성에 한하여 국방의 의무를 다할 수 있도록, 여성사병을 모집□ 안녕하세요, 저는 대한민국 국민, 26세 여성입니다. 우선 글을 쓰기에 앞서 국방안보에 대한 어떠한 정당의 의견, 성별분쟁에도 치우치치 않은 대한민국을 사랑하는 한국민의 의견임을 알려드립니다. 국민 개개인의 수단과 방법은 모두 달라도, 나라를 사랑하는 마음은 같을 것이라 생각합니다.
- **Target**: 여성의 군 복무 희망을 존중하지만, 현재 병역 제도는 전투 중심의 징병제로 운영되고 있어 지원 여부와 관계없이 여성 사병 모집이 군 구조와 임무 수행에 어려움을 초래할 수 있습니다.

### Example 5 — 보건복지

- **Topic**: [F03] 상세불명의 치매 코드는 치매 국가 책임제에서 제외해야합니다.
- **Input**: 현재 간호사로써, 과거 요양병원 병동 간호사로 근무했습니다. 행려로 의료원에서 치료받고, 그 치료비를 대신 내어주고 환자를 병원으로 데리고 와 입원을 시킵니다. 6개월동안 병원에서 거의 무상으로 데리고 있으면서, 6개월뒤엔 보호 1종 혹은 보호 2종으로 만듭니다. 그리고 수급비가 나오면 일부는 병원비로 받고 나머지 수급비로는 환자가 병원내의 매점에서 간식을 사먹을 수 있도록 합니다. 그렇게 지속적으로 데리고 있기위해서 넣는 코드가 [F03] 상세불명의 치매 입니다.
- **Target**: [F03] 코드를 남용하는 사례가 존재하더라도, 이를 제도 전반에서 제외하는 것은 오히려 정당한 지원이 필요한 치매 환자들에게 불이익을 줄 수 있습니다. 문제는 코드의 존재가 아니라, 그 사용을 감시하고 관리하는 시스템의 개선이 필요합니다.

### Example 6 — 일자리

- **Topic**: 부당해고.부당노동행위 엄정 처벌에 대하여
- **Input**: .부당해고/부당노동행위 엄벌을 언급한것에 대해서 대통령께서 위 사항을 엄정대처하고 처벌강화하시라고 하셨습니다. 저는 중소사업장과 비정규직조직화를 위해 활동하고 있습니다. 공단과 왕래가 많은 곳에 나가 홍보를 하면 노동조합가입하고 싶지만 못하는 이유는 해고라고 말합니다. "그거하면 짤리잖아요'하면서 대부분 말합니다. 즉, 제도적으로 아직 미약하니 강하게 처벌받지않으니 가입해도 소용없고 오히려 불이익만받는다고 생각합니다.
- **Target**: 부당해고와 부당노동행위에 대한 엄정한 처벌은 필요하지만, 처벌 강화만으로는 근본적인 해결이 어렵고, 노동자의 권리 인식 제고와 함께 제도적 지원 체계를 함께 마련해야 실질적인 변화가 가능합니다.

### Example 7 — 교통/건축/국토

- **Topic**: [링크수정본] 무단횡단 관련 과실 비율을 강력하게 개정해야합니다
- **Input**: 대한민국 무단횡단으로인한 매년 무단횡단 사망 자가 400명이 됍니다 OECD 3배나 됄정도로 심각합니다 1달전에도 어떤 운전자가 도로 달리다가 우산쓰고 무단횡단하던 보행자을 친 사고나 발생했습니다 근데 여기서 문제가 분명 보행자 과실 비율이 높아야하는데 오히려 운전자가 과실 비율이 더 높게 나옵겁니다 운전자 8 : 2 보행자 가 나왔습니다 이해가 안갔습니다 개인적인 생각으로 교통 약자인 보행자를 보호하는 것은 당연하지만 스스로 보호받을 권리를 포기한 무단 횡단자는 과실 비율 100%로 하고 벌금도
- **Target**: 무단횡단이 위험하다는 데 동의하지만, 운전자는 항상 주변 상황을 예측하고 사고를 방지할 책임이 있으며, 그 책임을 과실 비율 100%로 완전히 배제하는 것은 현실적인 안전 확보에 도움이 되지 않습니다.
