# Dataset examples

Random samples (seed=42) from each interim file after preprocessing.
These are the records that flow through merge_and_split into train/valid/test.

## IBM ArgQ 30K — real debate stance pairs (EN)

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

## mc-ai/conversation_dataset — persuasion dialogue (EN)

- File: `data/interim/en_mc_conversation.jsonl`
- Total records: 1,500

### Example 1

- **Topic**: Charity and Donation Request
- **Input**: I would be interested in doing do, however i must tell you i am generally very skeptical about charity orginizations in general. I know of many that i do not support due to the 6 figure salaries of the ceo. And the fact that not much of the donation actually get to the"cause".
- **Target**: I completely understand the skepticism due to a tenfold of shady organizations these days. However, i can guarantee you save the children is a professional organization that promises their donations go to a trustable fund to support international children's causes.
- **Meta**: `is_synthetic`=False · `conversation_id`=16444 · `corpus_id`=persuasionforgood

### Example 2

- **Topic**: Charity Donation Discussion
- **Input**: Ok i'm happy to at least get something. So out of your two dollar payment you are only willing to donate 10 cents?
- **Target**: I am not"only" willing. I am willing to donate 10 cents.
- **Meta**: `is_synthetic`=False · `conversation_id`=1195 · `corpus_id`=persuasionforgood

### Example 3

- **Topic**: Charity and Donations
- **Input**: Yes, they are both good charities. It has been very nice chatting with you.
- **Target**: Yes, nice chatting with you too!
- **Meta**: `is_synthetic`=False · `conversation_id`=8509 · `corpus_id`=persuasionforgood

## Isotonic/human_assistant_conversation — Q&A fluency (EN)

- File: `data/interim/en_isotonic_conversation.jsonl`
- Total records: 1,500

### Example 1

- **Topic**: casual conversation
- **Input**: What are the current trends in media marketing.
- **Target**: The current trends in media marketing include using social media to foster relationships with potential customers, leveraging data-driven decision-making, incorporating influencer marketing, utilizing artificial intelligence and machine learning for more customized marketing campaigns, and exploring new channels for reaching target audiences.
- **Meta**: `is_synthetic`=False

### Example 2

- **Topic**: casual conversation
- **Input**: I'm thinking about making a career change. What options should I consider.
- **Target**: Some potential career options you may want to consider include becoming a teacher, a nurse, an accountant, a computer programmer, a marketer, a web designer, an engineer, a lawyer, a financial analyst, a social worker, an entrepreneur, a carpenter, or a chef.
- **Meta**: `is_synthetic`=False

### Example 3

- **Topic**: casual conversation
- **Input**: What is the procedure for obtaining a passport.
- **Target**: The procedure for obtaining a passport requires that you first fill out an application form with your personal information and photograph. After submitting your application, you should be prepared to provide proof of identity, such as a driver's license, birth certificate or naturalization certificate. Once your application is approved, the passport will be mailed to you. It is important to note that you will also need to pay a passport fee before receiving your passport.
- **Meta**: `is_synthetic`=False

## SohamGhadge/casual-conversation — chit-chat (EN)

- File: `data/interim/en_casual_conversation.jsonl`
- Total records: 1,000

### Example 1

- **Topic**: casual conversation
- **Input**: i don't have money or a new car.
- **Target**: me, neither.
- **Meta**: `is_synthetic`=False

### Example 2

- **Topic**: casual conversation
- **Input**: you're watching too much tv.
- **Target**: what do you mean?
- **Meta**: `is_synthetic`=False

### Example 3

- **Topic**: casual conversation
- **Input**: how many invitations has she given out?
- **Target**: i really don't know, but i don't think she gave out that many yet.
- **Meta**: `is_synthetic`=False

## KLUE-NLI — Korean contradiction pairs (KO, no vLLM)

- File: `data/interim/ko_klue_nli.jsonl`
- Total records: 5,000

### Example 1

- **Topic**: 두 문장의 관점 차이
- **Input**: 가장 문제가 되는 것은 정보센터를 나가사키 항에 안 만들고 도쿄에 만든 것입니다.
- **Target**: 정보센터를 도쿄에 만든 것은 전혀 문제가 되지 않습니다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 2

- **Topic**: 두 문장의 관점 차이
- **Input**: 라이브커머스는 행사기간 마지막날에만 진행된다.
- **Target**: 라이브커머스는 각 지역행사장의 오픈스튜디오 또는 해당 지역 핫스팟에서 행사기간 동안 매일 진행된다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 3

- **Topic**: 두 문장의 관점 차이
- **Input**: 뉴욕패션위크 리한나 란제리쇼 사진을 모아봤다.
- **Target**: 리한나의 란제리 쇼 사진은 없다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

### Example 4

- **Topic**: 두 문장의 관점 차이
- **Input**: 2007년, 러시아의 블라디미르 푸틴 대통령이 대사관에 머무르던 당시에도 수류탄이 투척된 적이 있다.
- **Target**: 대사관에 수류탄이 투척된 적은 없다.
- **Meta**: `is_synthetic`=False · `conversion`=nli_contradiction_pair · `split`=train

## Korean Petitions — vLLM-synthesized rebuttals (KO, Qwen3-235B-A22B)

- File: `data/interim/ko_korean_petitions.jsonl`
- Total records: 9,992

Stratified by petition category for topic diversity. The source corpus is
heavily skewed to 소년/청소년 법 petitions (a hot 2017–2018 issue); those are
filtered out *here in the showcase only* — they remain in the training data.

### Example 1 — 안전/환경

- **Topic**: 자연을 깨끗이 합시다.
- **Input**: 여름철 해수욕장에 피서를 오고 나서 바닷가해변,바닷가 물속,계곡,산속 등에 쓰레기를 버리고 가는 사람들이 많은데요. 저는 제가 사랑하는 해변 한군데를 매년 자원봉사로 수중청소를 하고 있습니다. 모든 사람들이 참가했으면 합니다. 지금 무엇보다 시급한 문제가 북한도발에 대한 강력한 안보문제입니다. 유사시에 북한 김정은을 암살하기 위한 특수부대원을 키워놓게 지금부터 빨리 준비했으면 합니다. 저도 특수부대 나왔지만,실직적인 특수부대원을 만들었으면 합니다.
- **Target**: 자연 보호와 안보 강화는 서로 배타적인 문제가 아니라 함께 해결해야 할 과제입니다. 제가 사랑하는 해변을 지키는 것처럼, 국가를 지키기 위한 노력도 중요하지만 환경 보전은 지금 당장 우리 일상에서 실천할 수 있는 안보와 같은 필수적 책임입니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=안전/환경 · `num_agree`=1 · `begin`=2017-08-29 · `end`=2017-11-27

### Example 2 — 육아/교육

- **Topic**: 기간제교원의 정규직화를 반대합니다
- **Input**: 안녕하십니까 저는 경남의 공립단설유치원에서 근무하고 있는 교사입니다. 저는 2000년 졸업을 앞두고 공부를 하며 시험을 준비했는데 그 당시에 기간제교원이 정규직교사가 되면서 경남에서 뽑는 교원이 0명이었습니다 정말 충격이었습니다 그래서 어쩔수없이 경기도에 지원을 했고 3년근무한뒤 경남으로 내려올수 있었습니다 그런데 17만년에 또이런일이 생기다니요 제가 그당시에 겪었기에 지금 자신의 삶을 포기한채 2~3년 4~5년을 공부에 매달려있는 사람들은 어떤 심정일지 알것 같습니다 시험과 자격검정절차를 무의미
- **Target**: 기간제 교원의 정규직 전환은 교육 현장의 안정성과 지속성을 높이기 위한 것이며, 이는 학생들의 권리 보장과도 직결됩니다. 시험 준비의 어려움을 겪은 분들의 사정도 이해하지만, 교직 내 불안정한 고용 구조를 해소하는 것도 중요한 사회적 과제입니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=육아/교육 · `num_agree`=4 · `begin`=2017-08-29 · `end`=2017-11-27

### Example 3 — 정치개혁

- **Topic**: 부산여중생폭행사건 이대로 끝나게 하지 말아주세요.
- **Input**: 소년법 폐지해주시고 가해자들 법에맞게 처벌해주세요. 국민의 청원을 부디 무시하지 말아주세요.
- **Target**: 소년법 폐지는 성급한 해결책일 수 있으며, 철저한 재발 방지 대책과 교육적 접근이 함께 고려되어야 합니다. 국민의 목소리를 무시하는 것이 아니라, 보다 균형 잡힌 사회적 합의가 필요합니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=정치개혁 · `num_agree`=5 · `begin`=2017-09-05 · `end`=2017-12-04

### Example 4 — 외교/통일/국방

- **Topic**: 여성도 의무화적 군대 가야합니다
- **Input**: 여성도 의무적으로 군대를 가야됩니다 남성은 의무적이라 안가고 싶어도 가는 사람이 많지만 되려 여자들은 가고싶지만 대학을 못나왔다는 이유로 또 한 신체적 조건이 안된다는 이유로 차별 당하고 있습니다 언제까지 남녀차별로 살아가야 된다는 말입니까 대학을 못나와 취업을 못하게 하는 이 세상에서 군대에 가고 싶은 어려운 사람들은 그렇게 가지 못하고 그저 하늘에 별 따기 처럼 바라만 보고 있습니다 부디 청와대분들과 대통령님께서는 97년생으로 부터 여자들을 군대에 보내주시기 간곡히 부탁드리며 저는 가정 형편상
- **Target**: 여성의 군대 복무 의무화는 성별에 따른 신체적 차이와 사회적 여건을 고려하지 않은 일률적인 접근이 될 수 있으며, 대신 다양한 대체 복무 제도를 통해 형평성을 모색하는 것이 더 현실적입니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=외교/통일/국방 · `num_agree`=2 · `begin`=2017-09-03 · `end`=2017-12-02

### Example 5 — 보건복지

- **Topic**: 건강보험 가입자격과 외국인 문제....
- **Input**: 우연히 기사를 보고, 너무도 어처구니없는 현재의 건강보험 가입 규정에 대해 글을 올립니다. 한국의 시민들은 직장을 다니면서 임금을 받으며, 본인이 원하지 않아도 이 건강보험료는 임금에서 강제적으로 공제가 됩니다. 실직상태에서도 지역건강보험공단에서 청구서가 매월 날아오고, 우리들은 납부를 해야만 하죠. 그런데, 한국에 와서 겨우 3개월만 지나면 외국인들도 쉽게 건강보험 가입이 되고 그 이후에는 비싼 수입약도 내국인과 똑같이 100 % 건강보험 급여를 적용받을 수가 있군요.
- **Target**: 외국인의 건강보험 가입은 단순히 체류 기간만으로 결정되는 것이 아니라, 상당 기간 고용·납부 의무를 이행한 후에야 동등한 혜택을 받을 수 있도록 되어 있으며, 이는 사회적 연대와 포용의 원칙에 기반한 제도입니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=보건복지 · `num_agree`=2 · `begin`=2017-09-17 · `end`=2017-10-17

### Example 6 — 일자리

- **Topic**: '지역인재' 수정 및 블라인드 관련 청원
- **Input**: 지역인재와 관련된 청원이 있어서 글 올립니다. 1. 블라인드 채용을 표방하는 데 지역인재 우대 제도는 폐지하는 게 맞지 않습니까? 이게 '진짜' 블라인드인가요? 2. 블라인드와 지역인재 제도를 병행하는 것까지 양보한다 해도, 현재 공기업 등에서 진행하는 '지역인재'는 수정이 필요합니다. 현재 국가에서 '지역인재'라고 워딩을 해놓은 제도는 제대로 된 지역인재가 아니라고 생각합니다.
- **Target**: 지역인재 제도는 블라인드 채용의 취지를 해치기보다, 지역 간 불균형을 해소하려는 보완적 정책으로 병행될 수 있으며, 이를 통해 실질적인 기회 균등이 가능해질 수 있습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=일자리 · `num_agree`=420 · `begin`=2017-09-19 · `end`=2017-10-19

### Example 7 — 교통/건축/국토

- **Topic**: 시민의 눈물에 등 돌린 복기왕 아산시장
- **Input**: 안녕하십니까. 대통령님께 이 내용이 전달되기를 바라는 간절한 마음으로 글을 적습니다. 저는 충청남도 아산시에 거주하고 있는 30대 남자입니다. 저는 2년 전, 충청남도 아산시 풍기동에 새로 들어설 예정인 <이지더원> 이라는 아파트 29평형을 2억 3천 3백만원에 계약했습니다. 초등학교를 품은 아파트, 아산의 강남, 차 없는 단지 등 메리트가 확실해 보이는 아파트였기에 넉넉하지 않은 형편에도 무리를 무릅쓰고 계약을 했었습니다. 분양이 마감된 이후부터, 공사가 조금씩 진전되는 것이 보였습니다.
- **Target**: 아파트 분양은 개인의 선택과 계약에 기반한 사안으로, 시장이 모든 개별 민원에 개입할 수는 없습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=교통/건축/국토 · `num_agree`=332 · `begin`=2017-08-28 · `end`=2017-09-27
