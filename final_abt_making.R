library(tidyverse)
library(lubridate)


claim <- read_csv("claim_0702.csv")
policy <- read_csv("policy_0702.csv")
train <- read_csv("training-set.csv") %>% mutate(note = 'train')
test <- read_csv("testing-set.csv") %>% mutate(note = 'test')


#data format
claim %>% summary()
claim <- claim %>% 
  mutate(Accident_yr = str_sub(Accident_Date, 1,4) %>% as.integer(),
         Accident_mo = str_sub(Accident_Date, 6,7) %>% as.integer(),
         Driver_age = str_sub(DOB_of_Driver, 4,7) %>% as.integer(), Driver_age = Accident_yr - Driver_age)
cate_list_c <- c("Nature_of_the_claim","Driver's_Gender","Driver's_Relationship_with_Insured","Marital_Status_of_Driver",
                 "Cause_of_Loss", "Coverage","Claim_Status_(close,_open,_reopen_etc)","Accident_area")

claim[cate_list_c] <- lapply(claim[cate_list_c], factor)


policy %>% summary()
policy$iage <- NA
policy$dage <- NA
policy[!is.na(policy$ibirth),]$iage <- abs(policy[!is.na(policy$ibirth),]$ibirth %>% str_sub(4,7) %>% as.integer() - 2016)
policy[!is.na(policy$dbirth),]$dage <- abs(policy[!is.na(policy$dbirth),]$dbirth %>% str_sub(4,7) %>% as.integer() - 2016)
policy <- policy %>%
  mutate(Cancellation = ifelse(is.na(Cancellation), "N", Cancellation),
         car_age = 2016 - Manafactured_Year_and_Month)

cate_list_p <- c("Vehicle_Make_and_Model1","Vehicle_Make_and_Model2","Imported_or_Domestic_Car","Coding_of_Vehicle_Branding_&_Type",
                 "fpt", "Main_Insurance_Coverage_Group", "Insurance_Coverage","Distribution_Channel","fassured",
                 "fsex", "fmarriage", "aassured_zip", "iply_area", "fequipment1","fequipment2","fequipment3",
                 "fequipment4", "fequipment5", "fequipment6", "fequipment9", "nequipment9","Cancellation")

policy[cate_list_p] <- lapply(policy[cate_list_p], factor)


#### making model ####
"來做abt唷"
tmp <- train %>%
  bind_rows(test)

#排除原保費達20萬以上者
a <- policy %>%
  filter(Insured_Amount3 != 0) %>%
  group_by(Policy_Number) %>%
  summarise(prem_sum = sum(Premium)) %>%
  right_join(tmp) %>%
  filter(note == 'train' & prem_sum >= 200000) %>%
  select(Policy_Number)
tmp <- tmp %>% anti_join(a)

#by地區 保費(分不同類型)
a <- policy %>%
  filter(Insured_Amount3 != 0) %>%
  group_by(Policy_Number, Main_Insurance_Coverage_Group) %>%
  summarise(premm = sum(Premium)) %>%
  ungroup()

b <- a %>% filter(Main_Insurance_Coverage_Group == '車責') %>%
  select(-Main_Insurance_Coverage_Group)

area_premm <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(b) %>%
  group_by(iply_area) %>%
  summarise(premm_Q2_area_lia = median(premm),
            premm_Q1_area_lia = quantile(premm, 0.25),
            premm_Q3_area_lia = quantile(premm, 0.75))

b <- a %>% filter(Main_Insurance_Coverage_Group == '車損') %>%
  select(-Main_Insurance_Coverage_Group)

area_premm <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(b) %>%
  group_by(iply_area) %>%
  summarise(premm_Q2_area_dmg = median(premm),
            premm_Q1_area_dmg = quantile(premm, 0.25),
            premm_Q3_area_dmg = quantile(premm, 0.75)) %>%
  full_join(area_premm)

b <- a %>% filter(Main_Insurance_Coverage_Group == '竊盜') %>%
  select(-Main_Insurance_Coverage_Group)

area_premm <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(b) %>%
  group_by(iply_area) %>%
  summarise(premm_Q2_area_burg = median(premm),
            premm_Q1_area_burg = quantile(premm, 0.25),
            premm_Q3_area_burg = quantile(premm, 0.75)) %>%
  full_join(area_premm)

area_premm[,-1] <- area_premm[,-1] %>% lapply(function(x) ifelse(is.na(x), 0, x)) %>% as.data.frame() %>% as.tibble()

a <- a %>% group_by(Policy_Number) %>% summarise(premm = sum(premm))
area_premm <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(a) %>%
  group_by(iply_area) %>%
  summarise(premm_Q2_area = median(premm),
            premm_Q1_area = quantile(premm, 0.25),
            premm_Q3_area = quantile(premm, 0.75)) %>%
  full_join(area_premm)

tmp <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(area_premm) %>%
  select(-iply_area) %>%
  right_join(tmp)

#by地區 出險率
area_claim <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  left_join(claim[,c(3,13)]) %>%
  mutate(is_claim = ifelse(is.na(Coverage),0, 1)) %>%
  select(-Coverage) %>%
  distinct() %>%
  group_by(iply_area) %>%
  summarise(area_claim = sum(is_claim), area_tot = n()) %>%
  mutate(area_claim_rate = area_claim / area_tot)

tmp <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(area_claim) %>%
  select(-iply_area) %>%
  right_join(tmp)

#by地區 重製成本
area_RC <- policy %>%
  distinct(Policy_Number, iply_area, Replacement_cost_of_insured_vehicle) %>%
  group_by(iply_area) %>%
  summarise(area_avg_RC = mean(Replacement_cost_of_insured_vehicle),
            area_Q1_RC = quantile(Replacement_cost_of_insured_vehicle, 0.25),
            area_Q2_RC = median(Replacement_cost_of_insured_vehicle),
            area_Q3_RC = quantile(Replacement_cost_of_insured_vehicle, 0.75))

tmp <- policy %>%
  distinct(Policy_Number, iply_area) %>%
  inner_join(area_RC) %>%
  select(-iply_area) %>%
  right_join(tmp)


#考量性別 年齡
#法人的人因係數相當於30-60歲男性
tmp <- policy %>%
  distinct(Policy_Number, fsex, iage, dage) %>%
  mutate(iage_gp = ifelse(is.na(iage), 'bt3060',
                          ifelse(iage< 20, "le20", 
                                 ifelse(iage < 25, 'bt2025',
                                        ifelse(iage < 30, 'bt2530',
                                               ifelse(iage < 60, 'bt3060',
                                                      ifelse(iage < 70, 'bt6070','mr70'))))))) %>%
  mutate(iage_gp = factor(iage_gp),
         fsex = as.character(fsex),
         fsex = (ifelse(is.na(fsex), '1', fsex) %>% factor())) %>%
  select(-iage, -dage) %>%
  right_join(tmp)


#車損/竊盜/責任判別
Insur_gpInsur_gp <- policy %>%
  mutate(Premium = ifelse(Insured_Amount3 == 0, 0, Premium)) %>%
  select(Policy_Number, Main_Insurance_Coverage_Group, Premium) %>%
  group_by(Policy_Number, Main_Insurance_Coverage_Group) %>%
  summarise(tot_insur_gp = sum(Premium), 
            max_insur_gp = max(Premium), min_insur_gp = min(Premium),
            Q1_insur_gp = quantile(Premium, 0.25), Q2_insur_gp = median(Premium), 
            Q3_insur_gp = quantile(Premium, 0.75), IQR_insur_gp = IQR(Premium), 
            sd_insur_gp = sd(Premium, na.rm = T), sd_insur_gp = ifelse(is.na(sd_insur_gp), 0, sd_insur_gp),
            mean_insur_gp = mean(Premium, na.rm = T), cnt_insur_gp = n())

#限定車責
names(Insur_gp)[3:12] <- names(Insur_gp)[3:12] %>% paste("車責", sep = "_")
tmp <- Insur_gp %>%
  filter(Main_Insurance_Coverage_Group == '車責') %>%
  select(-Main_Insurance_Coverage_Group) %>%
  right_join(tmp)

#限定車損
names(Insur_gp)[3:12] <- names(Insur_gp)[3:12] %>% str_replace_all("車責", "車損")
tmp <- Insur_gp %>%
  filter(Main_Insurance_Coverage_Group == '車損') %>%
  select(-Main_Insurance_Coverage_Group) %>%
  right_join(tmp)

#限定竊盜
names(Insur_gp)[3:12] <- names(Insur_gp)[3:12] %>% str_replace_all("車損", "竊盜")
tmp <- Insur_gp %>%
  filter(Main_Insurance_Coverage_Group == '竊盜') %>%
  select(-Main_Insurance_Coverage_Group) %>%
  right_join(tmp)

#共同補值
tmp[,2:31] <- tmp[,2:31] %>%
  lapply(function(x) ifelse(is.na(x),0, x))


#汽車製造年度
tmp <- policy %>%
  distinct(Policy_Number, car_age) %>% 
  mutate(car_mgf_gp = ifelse(car_age == 0, "yr0",
                             ifelse(car_age == 1, "yr1",
                                    ifelse(car_age == 2, "yr2",
                                           ifelse(car_age == 3, 'yr3', 'yr4+'))))) %>%
  mutate(car_mgf_gp = factor(car_mgf_gp)) %>%
  right_join(tmp)

#重製成本
tmp <- policy %>%
  distinct(Policy_Number, Replacement_cost_of_insured_vehicle) %>%
  right_join(tmp)

#車齡
tmp <- policy %>%
  distinct(Policy_Number, car_age) %>% 
  mutate(car_age_gp = ifelse(car_age == 0, "yr0",
                             ifelse(car_age == 1, "yr1",
                                    ifelse(car_age == 2, "yr2",'yr3+')))) %>%
  mutate(car_age_gp = factor(car_age_gp)) %>%
  right_join(tmp)

#做車責等級/係數/車體係數的處理
tmp <- policy %>%
  distinct(Policy_Number, lia_class, plia_acc, pdmg_acc) %>%
  right_join(tmp)

#做總保費的統計量 多少險種
prem <- policy %>%
  mutate(Premium = ifelse(Insured_Amount3 == 0, 0, Premium)) %>%
  group_by(Policy_Number) %>%
  summarise(tot_premium = sum(Premium), 
            max_premium = max(Premium), min_premium = min(Premium),
            Q1_premium = quantile(Premium, 0.25), Q2_premium = median(Premium), 
            Q3_premium = quantile(Premium, 0.75), IQR_premium = IQR(Premium), 
            std_premium = sd(Premium, na.rm = T), std_premium = ifelse(is.na(std_premium), 0, std_premium),
            mean_premium = mean(Premium, na.rm = T), tot_cov = n())
prem <- prem %>%
  mutate(min_premium = as.integer(min_premium))
tmp <- prem %>% right_join(tmp)

#是否繼承前保單
tmp <- policy %>%
  distinct(Policy_Number, Cancellation) %>%
  right_join(tmp)

#是否為新單
tmp <- policy %>%
  distinct(Policy_Number, Prior_Policy_Number) %>%
  mutate(is_new = ifelse(is.na(Prior_Policy_Number), 1, 0)) %>%
  select(-Prior_Policy_Number) %>%
  right_join(tmp)

#被保險人有幾台車
a <- policy %>%
  distinct(`Insured's_ID`, Vehicle_identifier) %>%
  count(`Insured's_ID`)

tmp <- policy %>%
  distinct(Policy_Number, `Insured's_ID`) %>%
  left_join(a) %>%
  select(-`Insured's_ID`) %>%
  rename(has_n_car = n) %>%
  right_join(tmp)

#車系代號
tmp <- policy %>%
  distinct(Policy_Number, Imported_or_Domestic_Car) %>%
  right_join(tmp)

#載客量and分組
tmp <- policy %>%
  distinct(Policy_Number, qpt) %>%
  mutate(qpt_gp = ifelse(qpt <= 2, '1-2',
                         ifelse(qpt == 3, '3',
                                ifelse(qpt <=9, '4-9','10+')))) %>%
  mutate(qpt_gp = factor(qpt_gp)) %>%
  right_join(tmp)

#排氣量and分組
tmp <- policy %>%
  distinct(Policy_Number, `Engine_Displacement_(Cubic_Centimeter)`) %>%
  rename(ED = `Engine_Displacement_(Cubic_Centimeter)`) %>%
  mutate(ED_gp = ifelse(ED < 1000, "1-L",
                        ifelse(ED <=1400, '1-1.4L',
                               ifelse(ED <= 1800, '1.4-1.8L',
                                      ifelse(ED <= 2000, '1.8-2L',
                                             ifelse(ED <= 2500, '2-2.5L',
                                                    ifelse(ED <= 3000, '2.5L-3L',
                                                           ifelse(ED <= 4000, '3-4L', "4+L")))))))) %>%
  mutate(ED_gp = factor(ED_go)) %>%
  right_join(tmp)


#退保數
tmp <- policy %>%
  distinct(Policy_Number, Insurance_Coverage, Coverage_Deductible_if_applied) %>%
  filter(Coverage_Deductible_if_applied < 0) %>%
  count(Policy_Number) %>%
  rename(withdraw_num = n) %>%
  right_join(tmp) %>%
  mutate(withdraw_num = ifelse(is.na(withdraw_num), 0, withdraw_num))

#同一車主前一年度非車險保單件數
tmp <- policy %>%
  distinct(Policy_Number, `Multiple_Products_with_TmNewa_(Yes_or_No?)`) %>%
  rename(MPwT = `Multiple_Products_with_TmNewa_(Yes_or_No?)`) %>%
  right_join(tmp)

#被保險人性質
tmp <- policy %>%
  distinct(Policy_Number, fassured) %>%
  mutate(fassured = as.character(fassured),
         fassured = ifelse(fassured == '6', '2', fassured),
         fassured = fassured %>% factor()) %>%
  right_join(tmp)

#基本車責之16G
tmp <- policy %>%
  filter(Insurance_Coverage == '16G', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount1, Insured_Amount3, Premium) %>%
  mutate(IC_16G_Times = Insured_Amount3 / Insured_Amount1) %>%
  rename(IC_16G_amt1 = Insured_Amount1,
         IC_16G_amt3 = Insured_Amount3,
         IC_16G_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:5] <- tmp[,2:5] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車責之16P
tmp <- policy %>%
  filter(Insurance_Coverage == '16P', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium) %>%
  rename(IC_16P_amt = Insured_Amount3,
         IC_16P_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:3] <- tmp[,2:3] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車責之29K
tmp <- policy %>%
  filter(Insurance_Coverage == '29K', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium) %>%
  rename(IC_29K_amt = Insured_Amount3,
         IC_29K_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:3] <- tmp[,2:3] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車責之29B
tmp <- policy %>%
  filter(Insurance_Coverage == '29B', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium) %>%
  rename(IC_29B_amt = Insured_Amount3,
         IC_29B_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:3] <- tmp[,2:3] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車責之18@
tmp <- policy %>%
  filter(Insurance_Coverage == '18@', Insured_Amount3 != 0) %>%
  select(Policy_Number,Insured_Amount1, Insured_Amount3, Premium) %>%
  rename(IC_18A_amt1 = Insured_Amount1,
         IC_18A_amt3 = Insured_Amount3,
         IC_18A_premm = Premium) %>%
  mutate(IC_18A_times = IC_18A_amt3 / IC_18A_amt1) %>%
  right_join(tmp)
#共同補值
tmp[,2:5] <- tmp[,2:5] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車責之12L
tmp <- policy %>%
  filter(Insurance_Coverage == '12L', Insured_Amount3 != 0) %>%
  select(Policy_Number,Insured_Amount1, Insured_Amount3, Premium) %>%
  rename(IC_12L_amt1 = Insured_Amount1,
         IC_12L_amt3 = Insured_Amount3,
         IC_12L_premm = Premium) %>%
  mutate(IC_12L_times = IC_12L_amt3 / IC_12L_amt1) %>%
  right_join(tmp)
#共同補值
tmp[,2:5] <- tmp[,2:5] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車責之15F
tmp <- policy %>%
  filter(Insurance_Coverage == '15F', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium) %>%
  rename(IC_12L_amt3 = Insured_Amount3,
         IC_12L_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:3] <- tmp[,2:3] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#土法煉鋼之車損險
#基本車損之04M
tmp <- policy %>%
  filter(Insurance_Coverage == '04M', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium, Coverage_Deductible_if_applied) %>%
  rename(IC_04M_amt = Insured_Amount3,
         IC_04M_premm = Premium,
         IC_04M_ddctb = Coverage_Deductible_if_applied) %>%
  right_join(tmp)
#共同補值
tmp[,2:4] <- tmp[,2:4] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#基本車損之05E
tmp <- policy %>%
  filter(Insurance_Coverage == '05E', Insured_Amount3 != 0) %>%
  select(Policy_Number,Insured_Amount1, Insured_Amount3, Premium) %>%
  rename(IC_05E_amt1 = Insured_Amount1,
         IC_05E_amt3 = Insured_Amount3,
         IC_05E_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:4] <- tmp[,2:4] %>%
  lapply(function(x) ifelse(is.na(x),0, x))


#土法煉鋼之竊盜險
#基本竊盜05N
tmp <- policy %>%
  filter(Insurance_Coverage == '05N', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium, Coverage_Deductible_if_applied) %>%
  rename(IC_05N_amt = Insured_Amount3,
         IC_05N_premm = Premium,
         IC_05N_ddctb = Coverage_Deductible_if_applied) %>%
  right_join(tmp)
#共同補值
tmp[,2:3] <- tmp[,2:3] %>%
  lapply(function(x) ifelse(is.na(x),0, x))
#補自負額比率(沒有買的人)
tmp <- tmp %>%
  mutate(IC_05N_ddctb = ifelse(is.na(IC_05N_ddctb), 30, IC_05N_ddctb),
         IC_05N_ddctb = factor(IC_05N_ddctb))

#基本竊盜之09@
tmp <- policy %>%
  filter(Insurance_Coverage == '09@', Insured_Amount3 != 0) %>%
  select(Policy_Number, Insured_Amount3, Premium) %>%
  rename(IC_09A_amt = Insured_Amount3,
         IC_09A_premm = Premium) %>%
  right_join(tmp)
#共同補值
tmp[,2:3] <- tmp[,2:3] %>%
  lapply(function(x) ifelse(is.na(x),0, x))

#出險金額
tmp <- claim %>%
  group_by(Policy_Number) %>%
  summarise(Paid_Loss_Amt = sum(Paid_Loss_Amount),
            paid_Expenses_Amount = sum(paid_Expenses_Amount)) %>%
  mutate(paid_tot = Paid_Loss_Amt + paid_Expenses_Amount) %>%
  right_join(tmp) %>%
  mutate(Paid_Loss_Amt = ifelse(is.na(Paid_Loss_Amt), 0, Paid_Loss_Amt),
         paid_Expenses_Amount = ifelse(is.na(paid_Expenses_Amount), 0, paid_Expenses_Amount),
         paid_tot = ifelse(is.na(paid_tot), 0, paid_tot))

#已出險
tmp <- claim %>%
  distinct(Policy_Number) %>%
  mutate(is_claim = '1') %>%
  right_join(tmp) %>%
  mutate(is_claim = ifelse(is.na(is_claim), '0', is_claim))
#記在同一人身上
a <- tmp %>%
  filter(is_claim == '1', fassured != '2') %>%
  inner_join(policy[,1:2]) %>%
  distinct(`Insured's_ID`) %>%
  mutate(is_claim_by_person = '1')
tmp <- tmp %>%
  select(Policy_Number, is_claim) %>%
  inner_join(policy[,1:2]) %>%
  left_join(a) %>%
  distinct() %>%
  mutate(is_claim_by_person = ifelse(is.na(is_claim_by_person), '0', is_claim_by_person)) %>%
  mutate(is_claim_by_person = factor(is_claim_by_person)) %>%
  right_join(tmp)
#記在法人車牌上
a <- tmp %>%
  filter(is_claim == '1', fassured == '2') %>%
  inner_join(policy[,c(1,5)]) %>%
  distinct(Vehicle_identifier) %>%
  mutate(is_claim_by_car = '1')
tmp <- tmp %>%
  select(Policy_Number, is_claim) %>%
  inner_join(policy[,c(1,5)]) %>%
  left_join(a) %>%
  distinct() %>%
  mutate(is_claim_by_car = ifelse(is.na(is_claim_by_car), "0", is_claim_by_car)) %>%
  mutate(is_claim_by_car = factor(is_claim_by_car)) %>%
  right_join(tmp)

tmp <- tmp %>% mutate(is_claim = factor(is_claim))

#by車 來看險種分類金額
a <- policy %>%
  filter(Insured_Amount3 != 0) %>%
  group_by(Policy_Number, Main_Insurance_Coverage_Group) %>%
  summarise(prem_tot = sum(Premium)) %>%
  spread(Main_Insurance_Coverage_Group, prem_tot)

a <- a %>%
  ungroup() %>%
  lapply(function(x) ifelse(is.na(x), 0, x)) %>%
  as.data.frame() %>% as.tibble()

a <- policy %>%
  filter(Insured_Amount3 != 0) %>%
  distinct(Policy_Number,
           Vehicle_Make_and_Model1) %>%
  left_join(a) %>%
  select(-Policy_Number) %>%
  group_by(Vehicle_Make_and_Model1) %>%
  summarise(car_avg_車責 = mean(車責),
            car_avg_竊盜 = mean(竊盜),
            car_avg_車損 = mean(車損),
            car_Q2_車責 = median(車責),
            car_Q2_竊盜 = median(竊盜),
            car_Q2_車損 = median(車損),
            car_max_車責 = max(車責),
            car_max_竊盜 = max(竊盜),
            car_max_車損 = max(車損))

tmp <- policy %>%
  distinct(Policy_Number, Vehicle_Make_and_Model1) %>%
  left_join(car_ICgp) %>%
  select(-Vehicle_Make_and_Model1) %>%
  right_join(tmp)

#by車 出險狀況
car_claimcar_claim <- policy %>%
  select(Policy_Number, Vehicle_Make_and_Model1) %>%
  left_join(claim[,c(3,13)]) %>%
  mutate(is_claim = ifelse(is.na(Coverage),0, 1)) %>%
  select(-Coverage) %>%
  distinct() %>%
  group_by(Vehicle_Make_and_Model1) %>%
  summarise(claim = sum(is_claim), car_tot = n()) %>%
  mutate(car_claim_rate = claim / car_tot) %>%
  arrange(-car_claim_rate)

tmp <- policy %>%
  distinct(Policy_Number, Vehicle_Make_and_Model1) %>%
  left_join(car_claim) %>%
  select(-Vehicle_Make_and_Model1) %>%
  right_join(tmp)

#by車 基本狀況
car_profile <- policy %>%
  distinct(Policy_Number, Vehicle_Make_and_Model1, `Engine_Displacement_(Cubic_Centimeter)`,
           qpt, Replacement_cost_of_insured_vehicle) %>%
  group_by(Vehicle_Make_and_Model1) %>%
  summarise(car_avg_ED = mean(`Engine_Displacement_(Cubic_Centimeter)`),
            car_avg_qpt = mean(qpt),
            car_avg_Rcost = mean(Replacement_cost_of_insured_vehicle),
            car_Q2_ED = median(`Engine_Displacement_(Cubic_Centimeter)`),
            car_Q2_qpt = median(qpt),
            car_Q2_Rcost = median(Replacement_cost_of_insured_vehicle),
            car_Q1_ED = quantile(`Engine_Displacement_(Cubic_Centimeter)`, 0.25),
            car_Q1_qpt = quantile(qpt, 0.25),
            car_Q1_Rcost = quantile(Replacement_cost_of_insured_vehicle, 0.25),
            car_Q3_ED = quantile(`Engine_Displacement_(Cubic_Centimeter)`, 0.75),
            car_Q3_qpt = quantile(qpt, 0.75),
            car_Q3_Rcost = quantile(Replacement_cost_of_insured_vehicle, 0.75))
tmp <- policy %>%
  distinct(Policy_Number, Vehicle_Make_and_Model1) %>%
  left_join(car_profile) %>%
  select(-Vehicle_Make_and_Model1) %>%
  right_join(tmp)


#by通路 保額
a <- policy %>%
  filter(Insured_Amount3 != 0) %>%
  select(Policy_Number, Premium) %>%
  group_by(Policy_Number) %>%
  summarise(prem_tot = sum(Premium))

cnnl_prem <- policy %>%
  distinct(Policy_Number, Distribution_Channel) %>%
  left_join(a) %>%
  mutate(prem_tot = ifelse(is.na(prem_tot), 0, prem_tot)) %>%
  group_by(Distribution_Channel) %>%
  summarise(cnnl_avg_premm = mean(prem_tot), cnnl_cnt = n(),
            cnnl_Q2_premm = median(prem_tot),
            cnnl_Q1_premm = quantile(prem_tot, 0.25),
            cnnl_Q3_premm = quantile(prem_tot, 0.75))

tmp <- policy %>%
  distinct(Policy_Number, Distribution_Channel) %>%
  left_join(cnnl_prem) %>%
  select(-Distribution_Channel) %>%
  right_join(tmp)

#by通路 重製成本
cnnl_RC <- policy %>%
  distinct(Policy_Number, Distribution_Channel, Replacement_cost_of_insured_vehicle) %>%
  group_by(Distribution_Channel) %>%
  summarise(cnnl_avg_RC = mean(Replacement_cost_of_insured_vehicle),
            cnnl_Q1_RC = quantile(Replacement_cost_of_insured_vehicle, 0.25),
            cnnl_Q2_RC = median(Replacement_cost_of_insured_vehicle),
            cnnl_Q3_RC = quantile(Replacement_cost_of_insured_vehicle, 0.75))

tmp <- policy %>%
  distinct(Policy_Number, Distribution_Channel) %>%
  left_join(cnnl_RC) %>%
  select(-Distribution_Channel) %>%
  right_join(tmp)

#by通路 出險率
cnnl_claim <- policy %>%
  distinct(Policy_Number, Distribution_Channel) %>%
  left_join(claim[,c(3,13)]) %>%
  mutate(is_claim = ifelse(is.na(Coverage),0, 1)) %>%
  select(-Coverage) %>%
  distinct() %>%
  group_by(Distribution_Channel) %>%
  summarise(cnnl_claim = sum(is_claim), cnnl_tot = n()) %>%
  mutate(cnnl_claim_rate = cnnl_claim / cnnl_tot) %>%
  select(-cnnl_tot)

tmp <- policy %>%
  distinct(Policy_Number, Distribution_Channel) %>%
  left_join(cnnl_claim) %>%
  select(-Distribution_Channel) %>%
  right_join(tmp)


#共同變數 保費*年齡
a <- policy %>%
  distinct(Policy_Number, iage) %>%
  mutate(iage = ifelse(is.na(iage), median(iage, na.rm = T), iage))

tmp <- a %>%
  right_join(tmp) %>%
  mutate(AGExPremm = iage * tot_premium)

#排除萬分之一
tmp <- tmp %>%
  filter(Next_Premium < 10*10000)

