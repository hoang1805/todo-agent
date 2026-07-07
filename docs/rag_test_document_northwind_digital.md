# Test Questions and Ground Truth Answers

## Easy Questions

### Q1. What is the mission of Northwind Digital?

**Answer:** Northwind Digital’s mission is: "Make enterprise operations predictable, observable, and easy to improve."

### Q2. What is the flagship product of Northwind Digital?

**Answer:** The flagship product is ArborFlow.

### Q3. What are the two main ArborFlow environments?

**Answer:** The two main environments are Staging and Production.

### Q4. Which product is used as the observability portal?

**Answer:** Beacon is used as the observability portal.

### Q5. What is the default expiration time for file download tokens?

**Answer:** The default expiration time is five minutes.

### Q6. Which table stores image metadata records?

**Answer:** `tb_af_img_m` stores image metadata records.

### Q7. Does `tb_af_img_m` store the binary file itself?

**Answer:** No. It stores image metadata and references a file group through `file_gr_obid`.

### Q8. What are the initial supported image type codes?

**Answer:** `SYMBOL`, `DECAL`, `PACKAGE`, and `REFERENCE`.

### Q9. When is the full search index rebuild scheduled?

**Answer:** Every Sunday at 02:00 server time.

### Q10. What role can approve or reject library requests?

**Answer:** The Approver role can approve or reject library requests.

## Medium Questions

### Q11. Why should ArborFlow not use absolute file paths for image business logic?

**Answer:** Because file storage locations may change between environments. Image metadata should store `file_gr_obid` and use the File Component rather than relying on absolute paths.

### Q12. What should the frontend send for an empty array field?

**Answer:** It should send an empty JSON array `[]`, not an empty string `""`.

### Q13. When should a new image metadata record be created if a user types an unregistered image name in a multi-select input?

**Answer:** It should be created only when the user saves the form, not while the user is typing.

### Q14. What information should be shown in the connected object list on the Image Management detail screen?

**Answer:** It should show connected approved library objects and library requests, and clearly indicate the object type to avoid confusion.

### Q15. Why is querying by `library_obid` alone unsafe?

**Answer:** Because approved library objects use the composite key `library_obid + org_code`, so `library_obid` alone may not be globally unique.

### Q16. What is the recommended first action when image preview does not load?

**Answer:** Request a new preview token and check authorization logs.

### Q17. What should happen if a dynamic field type is unsupported?

**Answer:** The screen should not crash. The frontend should render an unsupported-field placeholder for administrators, while normal users should see a simplified read-only value if possible.

### Q18. Which values have priority when editing an existing object in Dynamic Property Rendering?

**Answer:** Previously saved object values have priority over all default sources.

### Q19. What are the four incident roles listed in the handbook?

**Answer:** Incident Commander, Technical Lead, Communications Lead, and Scribe.

### Q20. What is the communication cadence for a SEV-2 incident?

**Answer:** Updates should be sent every 30 minutes.

## Hard Questions

### Q21. Explain the correct flow for previewing an image on a detail screen.

**Answer:** The user opens a screen with a preview component. The frontend retrieves image metadata from ArborFlow API. If `file_gr_obid` exists, the frontend requests a preview token from the File Component. The File Component validates permission and compatibility. The frontend renders the preview using the tokenized URL. If the token expires, the frontend requests a new token.

### Q22. A user enters two new image names in the Library create form, one with leading spaces and one that duplicates an existing image name in the same type. What should the system do?

**Answer:** The system should trim leading and trailing spaces, reject empty names, and reject duplicate names within the same selected image type. It should create only valid new image records on save and connect them to the current library object or request.

### Q23. Why can the same display name appear more than once in image search results without necessarily being a bug?

**Answer:** The same image name can exist under different image types or different organizations. Also, a symbol and a decal may share the same display name. To determine duplication, the system must compare `img_nm`, `img_tp_cd`, and `org_code` together.

### Q24. A property-with-defaults API returns all category data and the browser crashes. What is the likely cause and correct behavior?

**Answer:** The likely cause is that the API returned too much category data instead of only the category needed for the current screen. The correct behavior is to return only the required category data, cache dataset configuration during the session, avoid rendering hidden fields, and optimize large field groups.

### Q25. During Atlas Sync, a database deadlock occurs. What retry behavior should be used?

**Answer:** Atlas Sync should retry database deadlocks up to 2 times with exponential backoff.

### Q26. During Atlas Sync, a validation failure occurs. Should the system retry automatically?

**Answer:** No. Validation failures should not be retried automatically.

### Q27. If a full search index rebuild fails, what should the system do?

**Answer:** It should continue using the previous index. If search results become stale for more than two hours, it should raise an SEV-2 incident.

### Q28. How should external upload APIs handle the Image Management data model?

**Answer:** They should follow the same structure as the internal UI. They must create image records and relationship rows using the same model, and they must validate image type, file extension, organization code, and authorization.

### Q29. In the image relationship model, can one image connect to multiple library objects, and can one library object have multiple images?

**Answer:** Yes. The `tb_af_img_obj_m` table supports many-to-many relationships, so one image may connect to multiple library objects, and one library object may have multiple images.

### Q30. If a screen requires immediate consistency after an update, should it rely only on the search index?

**Answer:** No. Search results may lag by up to five minutes, so screens requiring immediate consistency should read directly from the database through the API.

## Expert Questions

### Q31. A library request has images before approval. What should happen to image relationships when the request is approved?

**Answer:** The image relationships should be copied or reassigned according to the approval flow rules so that the approved library object retains the appropriate image connections.

### Q32. A user reports duplicate image names after saving new names from a multi-select input. Which fields should be checked before declaring it a duplicate data bug?

**Answer:** Check `img_nm`, `img_tp_cd`, and `org_code` together. The same name may be valid if it belongs to a different image type or organization.

### Q33. Why is it important for preview tokens to be scoped to user, organization, and file identifiers?

**Answer:** Scoping prevents a token issued for one user, organization, or file set from being reused to access unauthorized files. It enforces least privilege and protects cross-organization data boundaries.

### Q34. What is the safest way to handle unsupported uploaded files in Image Management?

**Answer:** Allow the File Component to store files if permitted by general file rules, but prevent unsupported files from being used as primary preview files. Show a clear validation message listing supported preview extensions.

### Q35. Why should new image records not be created while the user is typing in a multi-select input?

**Answer:** Because temporary or incomplete input could create unnecessary dirty data. Records should be created only when the form is saved.

### Q36. A frontend sends `""` for `selectedImageNames`, and the backend expects an array. What is the recommended fix?

**Answer:** Fix the frontend payload so it sends `[]` for no selected names. Backend coercion may be possible, but the recommended solution is correct client-side typing.

### Q37. If the Image Management detail screen shows both library objects and library requests, why must it display the object type?

**Answer:** Because the same display name may exist as an approved library object and as a pending request. Showing the object type prevents users from confusing the two.

### Q38. What should be included in a post-incident review for SEV-1 and SEV-2 incidents?

**Answer:** Summary, customer impact, timeline, root cause, what went well, what went poorly, corrective actions, owners, and due dates.

### Q39. In Dynamic Property Rendering, what is the priority order when editing an existing object?

**Answer:** Previously saved object values have the highest priority over API-provided defaults, organization-level settings, user profile information, and static dataset defaults.

### Q40. What combination of sections would be useful to answer a question about why image preview fails after a file upload?

**Answer:** Sections 4, 7, and 12 are useful. Section 4 explains image preview behavior and validation, Section 7 explains File Component preview flow and security, and Section 12 lists common support causes and first actions.

---

## Additional RAG Stress-Test Prompts

Use these prompts to test retrieval, grounding, and refusal to over-assume.

1. "Find all places where organization code affects uniqueness or access control."
2. "Compare approved library objects and library requests."
3. "Explain why a newly typed image name should not be inserted immediately."
4. "What could cause browser crashes in dynamic property rendering?"
5. "List all token-related security requirements."
6. "What are the differences between preview and download behavior?"
7. "Which incident severity should be used when search becomes stale after a failed rebuild?"
8. "What should happen when the external API uploads an unsupported image type?"
9. "Explain how a many-to-many image relationship is represented."
10. "What information should support ask for during initial triage?"

---

## Expected Retrieval Challenges

A strong RAG system should:

- Retrieve the exact section that contains the answer.
- Avoid mixing library objects with library requests.
- Preserve conditional language such as "if compatible file exists" and "only when the user saves the form."
- Recognize that `file_gr_obid` references a file group and is not a raw file path.
- Understand that search index results may lag and are not always immediately consistent.
- Avoid inventing unsupported image extensions.
- Distinguish between operational metadata and customer file contents.
- Provide answers grounded in document text instead of general assumptions.

---

## End of Document

This document is fictional and designed for testing RAG systems. All company names, product names, table names, endpoint names, and operational details are synthetic unless otherwise stated.
