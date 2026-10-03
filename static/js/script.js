// ===================== Live range labels =====================
document.querySelectorAll('input[type="range"]').forEach((input) => {
  const out = document.querySelector(`output[for="${input.id}"]`);
  if (out) {
    out.textContent = input.value;
    input.addEventListener('input', () => (out.textContent = input.value));
  }
});

// ===================== Tabs (Model Insights) =====================
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ===================== Risk color map =====================
const RISK_COLORS = {
  Low: '#34D399',
  Medium: '#FBBF24',
  High: '#F87171',
};

// Gauge: semicircle from -90deg (left, low) to +90deg (right, high)
const RISK_ANGLE = { Low: -60, Medium: 0, High: 60 };
const RISK_ARC_FRACTION = { Low: 0.33, Medium: 0.66, High: 1.0 };

function animateGauge(riskClass) {
  const needle = document.getElementById('gaugeNeedle');
  const arc = document.getElementById('gaugeArc');
  const angle = RISK_ANGLE[riskClass];
  const color = RISK_COLORS[riskClass];
  const circumference = 283; // approx path length of the semicircle

  needle.style.transform = `rotate(${angle - 90}deg)`;
  needle.setAttribute('stroke', color);

  const offset = circumference * (1 - RISK_ARC_FRACTION[riskClass]);
  arc.style.stroke = color;
  arc.style.strokeDashoffset = offset;

  const label = document.getElementById('riskLabel');
  label.textContent = riskClass + ' Risk';
  label.style.color = color;
}

function renderProbBars(probabilities) {
  const container = document.getElementById('probBars');
  container.innerHTML = '';
  ['Low', 'Medium', 'High'].forEach((cls) => {
    const pct = probabilities[cls] ?? 0;
    const row = document.createElement('div');
    row.className = 'prob-bar-row';
    row.innerHTML = `
      <span class="name">${cls}</span>
      <div class="prob-bar-track"><div class="prob-bar-fill" style="background:${RISK_COLORS[cls]}"></div></div>
      <span class="pct">${pct.toFixed(1)}%</span>
    `;
    container.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector('.prob-bar-fill').style.width = pct + '%';
    });
  });
}

function renderFactors(factors) {
  const list = document.getElementById('factorsList');
  list.innerHTML = '';
  factors.forEach((f) => {
    const li = document.createElement('li');
    li.textContent = f;
    list.appendChild(li);
  });
}

// ===================== Form submit =====================
const form = document.getElementById('riskForm');
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const payload = {
    Age: document.getElementById('Age').value,
    Gender: document.getElementById('Gender').value,
    BMI: document.getElementById('BMI').value,
    Sleep_Hours: document.getElementById('Sleep_Hours').value,
    Stress_Level: document.getElementById('Stress_Level').value,
    Screen_Time_Hours: document.getElementById('Screen_Time_Hours').value,
    Water_Intake_Liters: document.getElementById('Water_Intake_Liters').value,
    Caffeine_Intake_mg: document.getElementById('Caffeine_Intake_mg').value,
    Physical_Activity_Hours: document.getElementById('Physical_Activity_Hours').value,
    Skipped_Meals_Per_Week: document.getElementById('Skipped_Meals_Per_Week').value,
    Prior_Migraine_Frequency_Monthly: document.getElementById('Prior_Migraine_Frequency_Monthly').value,
    Family_History: document.getElementById('Family_History').checked ? 1 : 0,
    Hormonal_Changes: document.getElementById('Hormonal_Changes').checked ? 1 : 0,
    Weather_Sensitivity: document.getElementById('Weather_Sensitivity').checked ? 1 : 0,
    Smoking: document.getElementById('Smoking').checked ? 1 : 0,
    Alcohol_Consumption: document.getElementById('Alcohol_Consumption').checked ? 1 : 0,
    Light_Sensitivity_Score: document.getElementById('Light_Sensitivity_Score').value,
    Noise_Sensitivity_Score: document.getElementById('Noise_Sensitivity_Score').value,
  };

  const btn = form.querySelector('.predict-btn');
  const originalText = btn.innerHTML;
  btn.innerHTML = '<span>Analyzing…</span>';
  btn.disabled = true;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.success) {
      document.getElementById('resultPlaceholder').style.display = 'none';
      document.getElementById('resultContent').style.display = 'block';
      animateGauge(data.risk_class);
      renderProbBars(data.probabilities);
      renderFactors(data.contributing_factors);
      document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      alert('Prediction error: ' + data.error);
    }
  } catch (err) {
    alert('Could not reach the prediction server.');
    console.error(err);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
});
