# Phan tich Cross-state trong du an ARC-BPO

## Ket luan nhanh

`Cross-state, unadjusted` va `Cross-state, adjusted` la hai nhan mo ta
**cach so sanh chosen va rejected o muc token/chunk**. Chung khong phai la hai
ten ham rieng trong code ARC-BPO hien tai.

- **Cross-state, unadjusted**: so sanh truc tiep hai token/chunk duoc sinh tu
  hai prefix khac nhau, khong bu cho chenh lech chat luong san co giua hai
  prefix.
- **Cross-state, adjusted**: van so sanh truc tiep qua hai prefix khac nhau,
  nhung them mot correction phu thuoc state. Trong repo, y tuong nay duoc giu
  trong cac nhanh legacy `Q_tbpo` va `A_tbpo`.
- **ARC-BPO**: khong dung ca hai cach tren. Moi chunk duoc match voi target
  one-sided cua chinh no tai chinh prefix cua no, vi vay khong can correction
  cross-state.

## 1. "State" trong language model la gi?

Voi mot preference pair:

- prompt: `x`;
- chosen response: `y^w`;
- rejected response: `y^l`.

Tai vi tri token `t`, hai response tao ra hai state, hay hai prefix:

```math
s_t^w = [x, y_{<t}^w], \qquad
s_t^l = [x, y_{<t}^l].
```

Token dang duoc xet la:

```math
a_t^w = y_t^w, \qquad a_t^l = y_t^l.
```

Ngay ca khi hai token co cung chi so `t`, thong thuong van co:

```math
s_t^w \ne s_t^l,
```

boi vi chosen va rejected da di theo hai quy dao sinh khac nhau. Day la ly do
cach ghep token cung chi so duoc goi la **cross-state comparison**.

Trong code legacy, [`compute_tbpo_loss_mask`](utils.py#L214) chi giu cac vi tri
ma ca chosen va rejected deu co token. Vi vay phep ghep chi chay den do dai cua
response ngan hon:

```math
T = \min(|y^w|, |y^l|).
```

## 2. Dai luong policy/reference log-ratio

Tai tung state, dat:

```math
r_t^w =
\log \frac{\pi_\theta(y_t^w \mid s_t^w)}
           {\pi_{\mathrm{ref}}(y_t^w \mid s_t^w)},
```

```math
r_t^l =
\log \frac{\pi_\theta(y_t^l \mid s_t^l)}
           {\pi_{\mathrm{ref}}(y_t^l \mid s_t^l)}.
```

Code tinh cac dai luong nay theo tung token trong
[`Q_tbpo_get_batch_logps`](loss/loss_utils.py#L341). Bien
`chosen_logps_margin` va `rejected_logps_margin` trong `trainers.py` chinh la
cac log-ratio tren.

Repo su dung chieu **rejected tren chosen** cho model ratio, nen mot
`log_R` nho hon the hien policy nghieng ve chosen manh hon.

## 3. Cross-state, unadjusted

### Dinh nghia

Phien ban unadjusted ghep token chosen va rejected tai cung vi tri `t`, sau do
dung truc tiep:

```math
\log R_t^{\mathrm{unadjusted}}
= \beta (r_t^l-r_t^w).
```

No tuong duong voi mot cross-state ratio co correction:

```math
w_t=1 \quad\text{hay}\quad \log w_t=0.
```

Sau do `log_R` co the duoc dua vao Bregman loss theo tung token, nhu cach
[`bregman_loss`](loss/loss.py#L10) tinh loss tren cac vi tri duoc mask.

### Van de cua phuong phap

Hai log-ratio dang duoc danh gia tai hai state khac nhau. Do do
`r_t^l-r_t^w` co the tron lan hai nguon chenh lech:

1. chat luong cua token hien tai;
2. chat luong continuation da khac nhau do toan bo prefix truoc token do.

Vi du, chosen prefix co the da rat tot truoc vi tri `t`. Token chosen tai `t`
khong nhat thiet la nguyen nhan chinh lam response tot hon, nhung objective
unadjusted van co the gan phan lon preference signal cho token nay.

He qua la gradient co the:

- over-credit token chosen;
- over-penalize token rejected;
- bien sai lech prefix thanh tin hieu ve local action;
- phu thuoc vao viec ghep token theo chi so va cat bo phan du cua response dai
  hon.

### Anh xa vao repo

Repo hien tai **khong co preset rieng** ten `cross_state_unadjusted`. Day nen
duoc hieu la mot baseline/ablation co tinh khai niem: lay cong thuc TBPO theo
token va bo `log_w` hoac `delta_kl`.

Can phan biet no voi cong thuc khong correction trong
`BPO_SBA_concatenated_forward` tai `trainers.py:666`. BPO-SBA cong tong
log-prob cua ca response truoc khi so sanh, nen do la **sequence-level**, khong
phai cross-state token comparison.

## 4. Cross-state, adjusted

### Dinh nghia tong quat

Phien ban adjusted van ghep hai token duoc sinh tai hai state khac nhau, nhung
them correction state-only:

```math
\log R_t^{\mathrm{adjusted}}
= \beta\left(r_t^l-r_t^w+c(s_t^l)-c(s_t^w)\right).
```

Hoac viet theo multiplicative weight:

```math
\log R_t^{\mathrm{adjusted}}
= \beta\left(r_t^l-r_t^w+\log w_t\right).
```

Correction chi phu thuoc vao prefix/state, khong phu thuoc truc tiep vao token
dang duoc so sanh. Muc dich cua no la tach chenh lech continuation value cua
hai prefix khoi chenh lech cua local token.

Trong repo co hai cach uoc luong correction.

### 4.1. TBPO-Q: learned state baseline

[`Q_tbpo_concatenated_forward`](trainers.py#L377) dung mot baseline head de tao
mot scalar cho hidden state tai moi vi tri:

```math
b_\phi(s_t^w), \qquad b_\phi(s_t^l).
```

Correction la:

```math
\log w_t^{(Q)} = b_\phi(s_t^l)-b_\phi(s_t^w).
```

Cong thuc trong code tai `trainers.py:452-460` la:

```python
log_w = b_rejected - b_chosen
log_R = beta * (rejected_logps_margin - chosen_logps_margin + log_w)
```

Truoc khi dung, code center `log_w` theo tung preference pair va clamp no vao
khoang cau hinh. Hidden state cua backbone duoc detach truoc khi dua vao
baseline head. [`BaselineHead`](baseline_head.py#L5) la mot linear layer hoac
MLP nho tra ve mot scalar cho moi vi tri.

Y nghia cua baseline la uoc luong gia tri/partition-function cua prefix. Neu
chosen prefix da tot hon rejected prefix, correction giam viec quy toan bo loi
the do cho token chosen hien tai.

Danh doi cua TBPO-Q:

- can them baseline head va optimizer/state checkpoint;
- baseline uoc luong sai co the dua bias vao objective;
- ton them bo nho va compute, du nho hon backbone.

### 4.2. TBPO-A: KL-based state correction

[`A_tbpo_concatenated_forward`](trainers.py#L504) khong dung learned baseline
head. No tinh KL tai tung prefix:

```math
K_t^w = D_{\mathrm{KL}}\left(
\pi_{\mathrm{ref}}(\cdot\mid s_t^w)
\|\pi_\theta(\cdot\mid s_t^w)
\right),
```

```math
K_t^l = D_{\mathrm{KL}}\left(
\pi_{\mathrm{ref}}(\cdot\mid s_t^l)
\|\pi_\theta(\cdot\mid s_t^l)
\right).
```

Correction va model ratio la:

```math
\log w_t^{(A)}=K_t^l-K_t^w,
```

```math
\log R_t^{(A)}=\beta(r_t^l-r_t^w+K_t^l-K_t^w).
```

Day la dung voi implementation tai `trainers.py:532-552`:

```python
delta_logps = rejected_logps_margin - chosen_logps_margin
delta_kl = rejected_position_kl - chosen_position_kl
log_R = beta * (delta_logps + delta_kl)
```

Trong code hien tai, `compute_kl(..., direction="ref_to_policy")` tinh
`KL(reference || policy)` tren toan bo vocabulary tai moi vi tri.

Danh doi cua TBPO-A:

- khong can value/baseline head rieng;
- phai tinh distribution-level KL tai moi state;
- correction phu thuoc vao do chinh xac cua policy/reference KL;
- van con token alignment va length-mismatch cua cross-state comparison.

## 5. Vi sao adjustment la can thiet?

Neu hai action duoc so sanh tai **cung mot state**, state-only baseline xuat
hien o ca hai ve va tu triet tieu:

```math
[Q(s,a_1)-b(s)]-[Q(s,a_2)-b(s)] = Q(s,a_1)-Q(s,a_2).
```

Nhung trong cross-state comparison:

```math
[Q(s_t^w,a_t^w)-b(s_t^w)]
-[Q(s_t^l,a_t^l)-b(s_t^l)],
```

hai baseline khong triet tieu vi `s_t^w != s_t^l`. Ban unadjusted ngam gia
dinh chenh lech nay bang 0. Ban adjusted uoc luong va dua no vao ratio.

Adjustment lam mo hinh hop ly hon ve mat state/action credit assignment, nhung
khong thay doi viec chosen va rejected van bi ghep truc tiep theo vi tri.

## 6. ARC-BPO khac hai phuong phap tren nhu the nao?

ARC-BPO chia moi response thanh cac semantic chunk rieng:

```math
y^w=(c_1^w,\ldots,c_m^w), \qquad
y^l=(c_1^l,\ldots,c_n^l).
```

Voi moi chunk, no tinh exact policy/reference chunk log-ratio:

```math
a_\theta(c_i)
=\beta\sum_{t\in c_i}
\left[
\log\pi_\theta(y_t\mid s_t)
-\log\pi_{\mathrm{ref}}(y_t\mid s_t)
\right].
```

Sau do tung chosen/rejected chunk duoc match voi target one-sided cua chinh no:

```math
a_\theta(c_i^w)\rightarrow\tau_i^w,
\qquad
a_\theta(c_j^l)\rightarrow\tau_j^l.
```

ARC-BPO khong tao phep so sanh dang
`S(s_i^w,c_i^w)-S(s_j^l,c_j^l)`. Vi vay no khong can:

- ghep token/chunk chosen voi rejected;
- cat hai response ve cung do dai;
- correction `w_t`;
- baseline/value head;
- KL correction theo tung state.

Dieu nay duoc noi ro trong [README](README.md#L13) va docstring cua
[`arc_bpo_pair_loss`](loss/loss.py#L59). Trong implementation, chosen va
rejected duoc tinh chunk log-ratio doc lap tai `trainers.py:738-749`, sau do
dua vao one-sided loss tai `trainers.py:751-764`.

Rang buoc ket noi cac one-sided target voi preference response-level la:

```math
\sum_i\tau_i^w-\sum_j\tau_j^l=\Delta^\star.
```

Do do, ARC-BPO van bao toan response-level preference margin ma khong can tao
cross-state pair cho tung token/chunk.

## 7. Bang so sanh

| Thuoc tinh | Cross-state, unadjusted | Cross-state, adjusted | ARC-BPO |
|---|---|---|---|
| Granularity | Token hoac unit | Token hoac unit | Semantic chunk |
| Ghep chosen voi rejected | Co | Co | Khong |
| Hai prefix co the khac nhau | Co | Co | Khong so sanh truc tiep |
| State correction | Khong | Co | Khong can |
| Cach correction trong repo | Khong co | Baseline Q hoac KL A | One-sided target |
| Can baseline head | Khong | TBPO-Q: co | Khong |
| Can per-state KL | Khong | TBPO-A: co | Khong |
| Xu ly response khac do dai | Cat theo response ngan hon | Cat theo response ngan hon | Cho phep so chunk khac nhau |
| Rui ro chinh | Tron prefix quality voi token quality | Sai so/noise cua correction | Chat luong chunking va target allocation |

## 8. Anh xa ten goi vao code

| Khai niem | Thanh phan gan nhat trong repo |
|---|---|
| Cross-state, unadjusted | Ablation TBPO voi `log_w = 0`; khong co preset rieng |
| Cross-state, adjusted bang Q baseline | `loss=Q_tbpo`, `Q_tbpo_concatenated_forward` |
| Cross-state, adjusted bang advantage/KL | `loss=A_tbpo`, `A_tbpo_concatenated_forward` |
| Sequence-level, unadjusted | `loss=BPO_SBA`; khong phai cross-state token method |
| One-sided, khong cross-state | `loss=arc_bpo`, `arc_bpo_pair_loss` |

## 9. Dien giai ngan gon

- **Unadjusted** hoi: "Token chosen co policy/reference ratio tot hon token
  rejected khong?" va bo qua viec hai token den tu hai lich su khac nhau.
- **Adjusted** hoi cung cau tren, nhung co gang tru anh huong cua hai lich su
  bang baseline hoac KL correction.
- **ARC-BPO** khong dat hai token/chunk vao cung mot phep so sanh. No hoi:
  "Moi chunk tai state cua chinh no da dat one-sided target duoc phan bo tu
  response-level preference margin chua?"

Day la khac biet cot loi ma README tom tat bang cum
"without comparing winner and loser chunks across different prefix states".
