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

## Korean topic-seeded debate synth (KO, Qwen3-235B-A22B)

- File: `data/interim/ko_debate_synth.jsonl`
- Total records: 5,880  (98 unique motions × ~60 records each)

Curated motions in `data/seeds/ko_debate_motions.txt`. For each motion the
LLM produces 30 PRO/CON pairs at temp=0.9; each pair becomes 2 records
(pro→con and con→pro). Samples below show one motion per category.

### Example 1 — 안락사 허용

- **Topic**: 안락사 허용
- **Input**: 말기 환자는 극심한 고통 속에서 존엄한 죽음을 선택할 권리를 가져야 하며, 안락사는 인간다운 삶의 종료를 보장하는 마지막 수단이다.
- **Target**: 안락사를 허용하면 일시적인 고통이나 우울증으로 판단력이 흐려진 환자가 되돌릴 수 없는 선택을 할 위험이 있으며, 이는 오히려 존엄성을 해칠 수 있다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 2 — 사형제 폐지

- **Topic**: 사형제 폐지
- **Input**: 사형제는 인간의 존엄성을 침해하며, 국가가 공식적으로 생명을 빼앗는 행위는 도덕적으로 정당화될 수 없습니다.
- **Target**: 사형제는 극악무도한 범죄에 대한 정의 실현 수단으로서, 피해자와 유가족의 권리를 무시할 수 없다는 점에서 존속되어야 합니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 3 — 기본소득 도입

- **Topic**: 기본소득 도입
- **Input**: 기본소득 도입은 모든 시민에게 안정적인 생계 보장을 제공함으로써 빈곤과 사회적 불평등을 완화할 수 있다.
- **Target**: 그러나 모든 국민에게 무조건 지급하는 기본소득은 재정 부담을 가중시켜 오히려 복지 예산의 축소를 초래할 수 있다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 4 — 동성결혼 합법화

- **Topic**: 동성결혼 합법화
- **Input**: 동성결혼의 합법화는 성적 지향에 관계없이 모든 시민이 평등하게 결혼의 권리를 누릴 수 있도록 보장하는 중요한 인권적 조치입니다.
- **Target**: 그러나 결혼 제도는 전통적으로 가족의 형성과 자녀 양육이라는 사회적 기능을 목적으로 해왔으며, 동성 커플의 결혼 합법화는 이러한 제도의 본질을 훼손할 수 있습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 5 — 원자력 발전 확대

- **Topic**: 원자력 발전 확대
- **Input**: 원자력 발전은 온실가스 배출이 적어 기후 변화 대응에 효과적이며, 안정적인 전력 공급이 가능하다.
- **Target**: 그러나 핵폐기물 처리와 사고 위험 등 안전 문제로 인해 장기적인 환경 피해 가능성이 크며, 이는 기후 혜택을 상쇄할 수 있다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 6 — 비만세 도입

- **Topic**: 비만세 도입
- **Input**: 비만세 도입은 건강한 생활습관을 장려하고 비만 관련 의료 비용 증가를 억제하는 데 효과적인 정책 수단이다.
- **Target**: 비만세는 개인의 건강 상태를 단순히 경제적 부담으로 전환하여 차별을 초래할 수 있으며, 오히려 의료 접근성을 저해할 위험이 있다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 7 — AI 의료 진단 사용 허용

- **Topic**: AI 의료 진단 사용 허용
- **Input**: AI 의료 진단은 방대한 의료 데이터를 신속하게 분석하여 조기 진단과 정밀 의료를 가능하게 하므로 환자의 생존율을 높일 수 있다.
- **Target**: 그러나 AI의 분석이 완전히 신뢰할 수 없다는 점에서, 잘못된 진단이 발생할 위험이 높아져 오히려 환자의 건강에 심각한 피해를 줄 수 있다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

### Example 8 — 개 식용 금지

- **Topic**: 개 식용 금지
- **Input**: 개는 오랫동안 인간의 동반자로 살아온 반려동물로서, 그 식용은 동물복지의 관점에서 정당화될 수 없다.
- **Target**: 반려동물과 식용 동물을 구분하는 것은 문화적 맥락에 달려 있으며, 일부 전통을 일괄적으로 금지하는 것은 문화적 다양성을 존중하지 않는 태도다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=pro · `target_stance`=con · `conversion`=topic_seeded_pair

## Korean Petitions — vLLM-synthesized rebuttals (KO, Qwen3-235B-A22B)

- File: `data/interim/ko_korean_petitions.jsonl`
- Total records: 9,992

Stratified by petition category for topic diversity. The source corpus is
heavily skewed to 소년/청소년 법 petitions (a hot 2017–2018 issue); those are
filtered out *here in the showcase only* — they remain in the training data.

### Example 1 — 안전/환경

- **Topic**: 길거리에서 내가 담배냄새를 느끼고 담배를 꺼달라고 했을때 거부시 벌금을 물게 해주세요
- **Input**: 말 그대로입니다. 피고 남은 냄새의 잔재는 어쩔 수 없지만 사람이 많은 길거리나 가게 앞, 가게 안, 공공장소 등등 사람이 있는 곳에서 담배를 피우는데 피는 사람은 자유인데 그것을 강제로 맞는 사람은 거부하는 자유는 없습니까? 현 시대에서 담배는 명백하게 사회악입니다. 단속이 안된다는 점에서 마약보다도 더 성질이 나쁜 쓰레기이며 지구의 환경에도 매우 심각한 타격을 주는데 대체 왜 담배가 단속이 안되는지 모르겠습니다.
- **Target**: 담배 냄새에 대한 불편함은 충분히 이해하지만, 이미 금연 구역을 제외한 곳에서 흡연하는 것은 법적으로 허용된 개인의 선택이며, 그로 인한 미세한 냄새까지 규제하는 것은 과도한 간섭이 될 수 있습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=안전/환경 · `num_agree`=8 · `begin`=2017-09-12 · `end`=2017-12-11

### Example 2 — 육아/교육

- **Topic**: 학적부 기재관리 관련하여 형사입건, 처벌
- **Input**: 며칠전 뉴스입니다. 【 앵커멘트 】 서울의 한 고등학교 경시 대회에서 한 학생에게 수상 실적을 몰아준 일이 발생했습니다. 이런 수상 실적은 학생부종합전형에서 중요한 고려 요소입니다. 임성재 기자의 단독 보도입니다. 【 기자 】 서울의 한 고등학교입니다. 지난해, 두 개 교내 경시 대회에서 한 학생에게 상을 몰아준 정황이 포착됐습니다. 당시 진학 지도 담당 교사는 학생들이 진로를 결정하기 위해 스스로 어떤 목표를 세우고 노력했는지를 평가하는 대회에서 한 학생에게 특혜를 줬습니다.
- **Target**: 학생 한 명에게 상을 몰아준 정황이 보도되었지만, 이는 개별 교사의 부적절한 판단일 수 있으며, 학적부 기재 관리 전반에 대한 형사처벌로 이어져야 할 범죄적 행위임을 직접적으로 보여주진 않습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=육아/교육 · `num_agree`=0 · `begin`=2017-09-18 · `end`=2017-10-18

### Example 3 — 정치개혁

- **Topic**: 소수의 의견은 전달 되지 않는 것인지요? 청원을 보시는 모든 분들께 동의르 구합니다.
- **Input**: 청원 화면에 '일정 수준 이상의 추천을 받고 국정 현안으로 분류된 청원에 대해, 가장 책임있는 정부 및 청와대 당국자(장관, 대통령 수석비서관 등)의 답변 을 받을 수 있습니다. '라고 명시 되어 있습니다. 일정이상의 추천이란? 구체적으로 얼마의 추천의 의미하는지 궁급합니다. 이는 해석하기에 따라 소규모 모임은 상대적으로 참여 인원이 적기 때문에, 기회의 평등으로부터 멀어질 수 있다고 생각 합니다. 저희 전국노인요양공동생활가정연합회의 인원은 200명 미만 입니다. 그렇다면.
- **Target**: 청원 시스템은 일정 수의 추천을 기준으로 답변 대상을 선정하지만, 이는 인원 수보다는 사회적 공감과 관심의 정도를 반영하려는 목적을 가지고 있습니다. 소규모 단체라도 그 청원의 내용이 타당하고 널리 공감받을 수 있다면 충분히 목소리를 전달할 기회가 있습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=정치개혁 · `num_agree`=2 · `begin`=2017-08-28 · `end`=2017-09-27

### Example 4 — 외교/통일/국방

- **Topic**: 외교/북한 또질문입니다. 한중관계 개선위한 방법
- **Input**: * 추미애 더불어민주당 대표가 북한의 6차 핵실험 등 한반도 안보위기 국면과 관련해 "북한과 미국에 동시 특사를 파견해 북미-남북 간 투 트랙 대화를 추진하는게 어떻습니까? 추 대표는 열린 교섭단체 대표연설에서 "어느 순간 북·미 간 대화가 열리고 남북 간 대화가 열리는 장래를 준비해야 한다"고 전했습니다.
- **Target**: 북한과 미국에 동시에 특사를 파견하는 제안은 대화 재개를 위한 의지로 볼 수 있지만, 북한의 핵실험 후에도 명확한 비핵화 조치 없이 대화를 추진하면 제재의 실효성이 약화될 수 있습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=외교/통일/국방 · `num_agree`=0 · `begin`=2017-09-09 · `end`=2017-09-16

### Example 5 — 보건복지

- **Topic**: 임시 공휴일 및 법정 공휴일은 누구만을 위한 휴일입니까?
- **Input**: 현재 저는 중소 병원에서 근무중입니다. 제가 다니는 병원은 주 5일제를 가장 하여 평일 하루, 일요일 하루 이런식으로 주휴 약휴가 존재하고 설,추석 및 근로자의 날 이외의 공휴일은 전혀 보장 받지 못하고 오히려 공휴일에는 더욱 출근을 강요하고 있습니다. 그나마 저는 나은 편이라고 생각하는데 일반 중소기업은 야근 강요는 물론 이고 휴일에 출근하여 추가 수당 또한 지급 받지 못하는게 현재 우리나라의 실정입니다.
- **Target**: 공휴일은 모든 근로자가 공평하게 향유할 수 있어야 하는 권리이며, 의료 현장과 같은 필수 업종에 종사하는 분들의 희생을 전제로 공휴일을 무시해서는 안 됩니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=보건복지 · `num_agree`=2 · `begin`=2017-09-02 · `end`=2017-12-01

### Example 6 — 일자리

- **Topic**: 블라인드 채용의 역차별 개선
- **Input**: 최근 일자리와 관련하여 정부에서 지속적으로 관심을 가져준 덕분인지 대기업이나 공공기관에서 긍정적인 신호를 보내고 있습니다. 특히 하반기부터 적용된 블라인드 채용으로 인해 양질의 취업처에 대한 균등한 기회를 보장해주기 위한 시도는 긍정적이라고 생각합니다. 하지만 국정농단으로 인한 어려움이 한창이던 올해 초부터 상대적으로 어려움에 처한 이들이 있습니다. 바로 특성화고에서 한창 취업을 준비중인 3학년 재학생들입니다.
- **Target**: 블라인드 채용이 균등한 기회를 제공하려는 점은 긍정적이지만, 특성화고 학생들이 기술과 실무 능력을 어필할 수 있는 경로를 제한함으로써 오히려 그들의 경쟁력을 외면하는 결과를 초래하고 있습니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=일자리 · `num_agree`=0 · `begin`=2017-09-15 · `end`=2017-10-15

### Example 7 — 교통/건축/국토

- **Topic**: 자전거 도로교통법 강화 부탁드립니다
- **Input**: 현장에서 잡지 못하면 영원히 처벌이 불가능한 자전거뻉소니! 자전거도 도로를 달리고싶으면 번호판을 의무화 해야합니다!
- **Target**: 번호판 의무화보다는 공익제보 앱 활성화와 단속 인력 확충으로 실질적인 단속을 강화하는 것이 더 현실적인 해결책입니다.
- **Meta**: `is_synthetic`=True · `synthesis_prompt_version`=v1 · `input_stance`=petition_position · `target_stance`=opposition · `category`=교통/건축/국토 · `num_agree`=1 · `begin`=2017-09-14 · `end`=2017-12-13
