# TODO List for Building Fully Functional Green Roof AI Web App

- [x] Enhance ML Logic in monitoring.py:
  - Remove JSON file generation; return dict directly.
  - Improve weed detection dummies for realistic random boxes based on image size.
  - Improve disease detection with random types (fungal, bacterial, viral) and masks.
  - Add automatic estimation of temperature and humidity from soil image (based on color analysis).
  - Add overall health score (0-100) calculation.
  - Update recommendations to be detailed and science-backed.

- [x] Update Backend in app.py:
  - Remove manual temperature/humidity input fields from form.
  - Save original uploaded images for side-by-side display.
  - Pass original and annotated image paths to results template.
  - Ensure no JSON export; all data passed directly.

- [x] Modernize UI in templates/index.html:
  - Update CSS for nature-themed (green/white), gradients, animations, card-style displays.
  - Add drag-and-drop upload functionality.
  - Add live progress bar and loading animation ("AI Analyzing...").
  - Remove temperature/humidity input fields.

- [x] Update templates/results.html:
  - Remove JSON display section.
  - Add recommendation summary panels: Weed Status, Disease Diagnosis, Soil Moisture & Environment Insights, Overall Roof Health Score.
  - Display side-by-side comparison of uploaded and annotated images.
  - Show all outputs visually with clear text + visual recommendations.

- [x] Create static/app.js:
  - Handle drag-and-drop file uploads.
  - Show progress bar during form submission.
  - Add loading spinner/animation for smooth interactions.

- [x] Update requirements.txt if needed (ensure all libs are present).

- [ ] Test the app locally with sample images.
- [ ] Verify realistic outputs and visual display.
- [ ] Ensure responsive design.
